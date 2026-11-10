"""Seeded synthetic call generator for the Booking Forensics demo.

Produces realistic (templated + varied) inbound call transcripts for three
fictional home-services businesses, each with a booking outcome, plus the
counterfactual outcome of every lead under three proposed agent changes.

The world model is a small causal simulator: every call has hidden caller
traits (price sensitivity, urgency, flexibility, patience) and a fixed set of
random draws ("common random numbers"). Agent policy decides which act is used
at each decision point; each act carries a hazard that the caller leaves.
Replaying a call under a new policy re-runs the same caller with the same random
draws, so the counterfactual is exact inside the simulation.

Also simulated: an LLM judge that scores each replay (noisy and optimistically
biased toward the "fixed" behaviour). The judge is the thing the estimator has
to correct with a small set of human labels.

Only the Python standard library is used. Run:  python3 data/generate.py
"""
import json
import math
import os
import random

SEED = 20261030
DAYS = 30  # Oct 1 - Oct 30, 2026

CUSTOMERS = [
    {"id": "ridgeline", "name": "Ridgeline Heating & Air", "trade": "HVAC", "city": "Columbus, OH",
     "leads_per_day": 27, "deflect_price": 0.85, "offer_alt": 0.35, "triage": 0.55, "tight": 0.38,
     "dispatch_fee": 89, "emergency_fee": 149},
    {"id": "copperline", "name": "Copperline Plumbing", "trade": "Plumbing", "city": "Tampa, FL",
     "leads_per_day": 22, "deflect_price": 0.9, "offer_alt": 0.6, "triage": 0.8, "tight": 0.27,
     "dispatch_fee": 79, "emergency_fee": 179},
    {"id": "brightwire", "name": "Brightwire Electric", "trade": "Electrical", "city": "Raleigh, NC",
     "leads_per_day": 16, "deflect_price": 0.4, "offer_alt": 0.2, "triage": 0.85, "tight": 0.5,
     "dispatch_fee": 99, "emergency_fee": 199},
]

# service: (label, avg job value $, P(emergency), problem phrases, urgent phrases)
SERVICES = {
    "HVAC": [
        ("Furnace / no heat", 410, 0.45,
         ["my furnace keeps shutting off", "the heat isn't kicking on", "the furnace is blowing cold air"],
         ["we have no heat at all and it's 31 degrees tonight", "there's no heat and I have a newborn at home",
          "no heat and my mom is 84, it's freezing in here"]),
        ("AC / heat pump repair", 450, 0.15,
         ["the heat pump is making a grinding noise", "the outdoor unit won't turn on", "the thermostat is blank"],
         ["the unit is smoking a little", "there's a burning smell coming from the vents"]),
        ("Tune-up / maintenance", 139, 0.0,
         ["I'd like a fall tune-up on the furnace", "I want to get the system checked before winter"], []),
        ("System replacement estimate", 3400, 0.0,
         ["we're thinking about replacing our 20 year old system", "I want a quote on a new furnace and AC"], []),
    ],
    "Plumbing": [
        ("Leak repair", 480, 0.5,
         ["there's a slow drip under the kitchen sink", "the toilet base is leaking"],
         ["water is coming through the ceiling right now", "a pipe burst and there's water all over the floor"]),
        ("Drain clog", 260, 0.2,
         ["the kitchen drain is backing up", "the shower drains really slow"],
         ["sewage is backing up into the tub"]),
        ("Water heater", 1850, 0.35,
         ["the water heater is making popping sounds", "we only get lukewarm water"],
         ["the water heater is leaking all over the garage", "we have no hot water and it's leaking"]),
        ("Sewer camera inspection", 650, 0.0,
         ["we're buying a house and want the sewer line scoped", "I want a camera inspection on the main line"], []),
    ],
    "Electrical": [
        ("Panel upgrade", 2600, 0.0,
         ["I need a panel upgrade for a new EV", "our panel is a Federal Pacific and the inspector flagged it"], []),
        ("Outlet / breaker repair", 310, 0.35,
         ["a breaker keeps tripping in the kitchen", "half the outlets in the living room stopped working"],
         ["an outlet sparked and now there's a burning smell", "the breaker box is buzzing and warm to the touch"]),
        ("EV charger install", 1450, 0.0,
         ["I want a Level 2 charger installed in the garage", "we just got a Tesla and need a charger put in"], []),
        ("Generator estimate", 2200, 0.0,
         ["we want a quote on a whole-home generator", "I'm interested in a standby generator"], []),
    ],
}

FIRST = ["Maria", "James", "Linda", "Robert", "Priya", "Michael", "Jennifer", "David", "Aisha", "Carlos", "Karen",
         "Brian", "Nicole", "Kevin", "Sandra", "Tom", "Megan", "Luis", "Rachel", "Greg", "Denise", "Omar", "Holly",
         "Wes", "Tanya", "Frank", "Grace", "Hector", "Julie", "Ray"]
LAST = ["Johnson", "Patel", "Nguyen", "Garcia", "Miller", "Brooks", "Okafor", "Kim", "Sullivan", "Reyes", "Turner",
        "Hughes", "Bennett", "Foster", "Ramirez", "Coleman", "Price", "Ward", "Ellis", "Hayes"]
STREETS = ["Maple Ridge Dr", "Oak Hollow Ln", "Briarwood Ct", "Lakeview Ave", "Cedar Bend Rd", "Willow Creek Way",
           "Harbor Point Blvd", "Sycamore St", "Foxfield Dr", "Pinecrest Rd", "Stonegate Ct", "Meadowlark Ln"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
WINDOWS = ["8 to 11 in the morning", "11 to 2", "2 to 5 in the afternoon"]


def logistic(x):
    return 1.0 / (1.0 + math.exp(-x))


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


# ---------------------------------------------------------------- policies
BASE = {"id": "baseline"}
CANDIDATES = [
    {"id": "quote_fee", "name": "Quote the diagnostic fee when asked about price",
     "short": "Quote diagnostic fee",
     "change": "When a caller asks what it costs, state the dispatch/diagnostic fee and that it is credited toward the repair, instead of deferring pricing to the technician.",
     "targets": ["PRICE_DEFLECTED"]},
    {"id": "offer_alts", "name": "Offer the two nearest open windows + waitlist",
     "short": "Offer alternatives + waitlist",
     "change": "When the requested window is full, proactively offer the two nearest open windows and add the caller to the cancellation waitlist.",
     "targets": ["NO_ALTERNATIVE_SLOT"]},
    {"id": "emergency_triage", "name": "Escalate no-heat / active-leak / sparking to same-day",
     "short": "Same-day emergency escalation",
     "change": "Treat no heat, active water leaks, sparking or burning smells as emergencies: offer same-day dispatch (with the emergency fee) instead of the next routine opening.",
     "targets": ["EMERGENCY_TRIAGED_ROUTINE"]},
]


def policy_value(cust, policy, key):
    """Probability that the agent takes the 'good' act under a policy."""
    if key == "quote":
        return 1.0 if policy == "quote_fee" else 1 - cust["deflect_price"]
    if key == "alt":
        return 1.0 if policy == "offer_alts" else cust["offer_alt"]
    if key == "triage":
        return 0.97 if policy == "emergency_triage" else cust["triage"]
    raise KeyError(key)


def simulate(call, cust, policy):
    """Run the decision process. Returns (booked, events, p_book, mode, mode_step).

    events: list of (step, act) in order; acts drive the transcript renderer.
    p_book: exact probability of booking given caller traits and the acts the
    policy chose (randomness only from the leave hazards).
    """
    u = call["u"]
    tr = call["traits"]
    ev = []
    surv = 1.0
    booked = True
    mode = None

    linger = {"left": 0}

    def hazard(step, act, h, m):
        """Append an act; draw the leave hazard. Returns True when the call ends.
        Some callers do not leave on the spot: they linger for 1-2 more steps
        and then end the call with a generic excuse (so the losing turn is not
        simply the last agent turn)."""
        nonlocal surv, booked, mode
        if not booked:  # lingering after the damage: no new hazards
            ev.append((step, act, False))
            linger["left"] -= 1
            return linger["left"] <= 0
        surv *= (1 - h)
        if u["leave_" + step] < h:
            booked = False
            mode = m
            if u["linger"] < 0.45 and step not in ("book", "close"):
                linger["left"] = 1 if u["linger"] < 0.3 else 2
                ev.append((step, act, "linger"))
                return False
            ev.append((step, act, True))
            return True
        ev.append((step, act, False))
        return False

    def plain(step, act):
        ev.append((step, act, False))
        if not booked:
            linger["left"] -= 1
            return linger["left"] <= 0
        return False

    emergency = call["emergency"]
    # 1. triage
    recognised = u["recognise"] < policy_value(cust, policy, "triage")
    false_alarm = (not emergency and call["mild_cue"] and policy == "emergency_triage"
                   and u["false_alarm"] < 0.35)
    if emergency and recognised or false_alarm:
        h = 0.08 + 0.35 * tr["price"] * (0.4 if emergency else 1.0)
        if hazard("triage", "EMERGENCY_ROUTE", h, "EMERGENCY_FEE_DECLINED"):
            return False, ev, 0.0, mode
    elif emergency:
        h = 0.30 + 0.45 * tr["urgency"]
        if hazard("triage", "ROUTINE_FOR_URGENT", h, "EMERGENCY_TRIAGED_ROUTINE"):
            return False, ev, 0.0, mode
    else:
        if plain("triage", "ROUTINE"):
            return False, ev, 0.0, mode
    # 2. price question
    if call["asks_price"]:
        if u["quote"] < policy_value(cust, policy, "quote"):
            h = 0.06 + 0.24 * tr["price"]
            # a stated fee is a different stimulus than a deflection, so the
            # caller's reaction is drawn independently (this is what creates
            # regressions: some callers who tolerated "the tech will quote you"
            # balk at "$89")
            if hazard("quote", "PRICE_QUOTE", h, "FEE_DECLINED"):
                return False, ev, 0.0, mode
        else:
            h = 0.62 * tr["price"] ** 1.2
            if hazard("price", "PRICE_DEFLECT", h, "PRICE_DEFLECTED"):
                return False, ev, 0.0, mode
    # 3. info capture
    if call["addr_loop"]:
        h = 0.08 + 0.35 * (1 - tr["patience"])
        if hazard("info", "ADDRESS_LOOP", h, "ADDRESS_CAPTURE_LOOP"):
            return False, ev, 0.0, mode
    else:
        if plain("info", "INFO_OK"):
            return False, ev, 0.0, mode
    # 4. scheduling (emergency route books same day, no slot problem)
    same_day = any(a == "EMERGENCY_ROUTE" for _, a, _f in ev)
    if not same_day:
        if call["slot_full"]:
            if u["alt"] < policy_value(cust, policy, "alt"):
                h = 0.05 + 0.25 * (1 - tr["flex"])
                if hazard("slot", "ALT_OFFERED", h, "ALTERNATIVE_DECLINED"):
                    return False, ev, 0.0, mode
            else:
                h = 0.25 + 0.5 * (1 - tr["flex"])
                if hazard("slot", "NO_ALT_SLOT", h, "NO_ALTERNATIVE_SLOT"):
                    return False, ev, 0.0, mode
        else:
            if plain("slot", "SLOT_OK"):
                return False, ev, 0.0, mode
    # 5. booking tool
    if call["tool_error"]:
        if hazard("book", "BOOKING_ERROR", 0.75, "BOOKING_TOOL_ERROR"):
            return False, ev, 0.0, mode
    # 6. shopping callers who never intended to commit
    if call["shopping"]:
        if hazard("close", "SHOPPER_CLOSE", 0.55, "CALLER_SHOPPING"):
            return False, ev, 0.0, mode
    if not booked:  # lingering caller reached the end of the flow
        return False, ev, 0.0, mode
    ev.append(("close", "CONFIRM", False))
    return True, ev, surv, None


# ---------------------------------------------------------------- transcript rendering
def render(call, cust, events, booked, mode, rng):
    """Render events into a [speaker, text] transcript. rng is a per-call RNG so
    the same call renders the same words for the same acts under any policy."""
    T = []
    A = lambda s: T.append(["agent", s])
    C = lambda s: T.append(["caller", s])
    name, svc = call["name"], call["service"]
    A(rng.choice([f"Thanks for calling {cust['name']}, this is Ava. How can I help you today?",
                  f"{cust['name']}, this is Ava speaking. What can I help you with?",
                  f"Hi, you've reached {cust['name']}. I can help get you scheduled. What's going on?"]))
    opener = call["problem"]
    if call["shopping"]:
        opener += rng.choice([". I'm just calling around to compare a few companies", ". I'm getting a couple of quotes"])
    if call["mild_cue"] and not call["emergency"]:
        opener += {"HVAC": ", and it's getting pretty cold in the house",
                   "Plumbing": ", and there's a little water on the floor",
                   "Electrical": ", and one of the outlets felt warm"}[cust["trade"]]
    C(rng.choice(["Hi, yeah, ", "Hey there, ", "Hi, ", "Yes hi, "]) + opener + ".")
    truth = {"turn": None}
    for step, act, fatal in events:
        lost_here = fatal is True
        lingering = fatal == "linger"
        n_before = len(T)
        if act == "EMERGENCY_ROUTE":
            A(rng.choice([
                f"I'm sorry, that sounds urgent. I can get a technician out to you today as an emergency visit. There is a ${cust['emergency_fee']} emergency dispatch fee. Would you like me to set that up?",
                f"Okay, let's treat that as an emergency. We can get someone out today; the same-day emergency visit has a ${cust['emergency_fee']} dispatch fee. Should I go ahead?"]))
            if lost_here:
                C(rng.choice([f"${cust['emergency_fee']} just to show up? No, I'll wait it out.",
                               "Oh, I didn't want to pay an emergency fee. Never mind, I'll figure something else out."]))
            else:
                C(rng.choice(["Yes, please, the sooner the better.", "Okay, yes, let's do that."]))
        elif act == "ROUTINE_FOR_URGENT":
            A(rng.choice(["I can help with that. I'll set it up as a standard repair visit; our openings are usually a few days out.",
                          "Got it. I can book a regular service appointment for that. We're typically scheduling a few days out right now."]))
            if lost_here:
                C(rng.choice(["A few days? I can't go that long. I'll call someone who can come today.",
                              "No, I need somebody today. I'll try another company, thanks."]))
            else:
                C(rng.choice(["Ugh, okay, I guess that will have to work.", "That's a while, but fine."]))
        elif act == "ROUTINE":
            A(rng.choice(["Sorry to hear that. I can get that scheduled for you.",
                          "Absolutely, we can help with that. Let me get a few details.",
                          "Okay, I can set up a visit for that."]))
        elif act == "PRICE_QUOTE":
            C(rng.choice(["Before we go further, how much is this going to cost?", "What do you charge to come out?",
                          "Can you tell me what the price is?"]))
            A(rng.choice([f"Our diagnostic visit is ${cust['dispatch_fee']}, and that's credited toward the repair if you go ahead with it.",
                          f"There's a ${cust['dispatch_fee']} service fee for the technician to diagnose it, and it gets applied to the repair."]))
            if lost_here:
                C(rng.choice([f"${cust['dispatch_fee']} just to look at it? I'll pass for now.",
                              "Hmm, that's more than I wanted to spend. Let me think about it."]))
            else:
                C(rng.choice(["Okay, that's fair.", "Alright, that works."]))
        elif act == "PRICE_DEFLECT":
            C(rng.choice(["Before we go further, how much is this going to cost?", "What do you charge to come out?",
                          "Can you tell me what the price is?"]))
            A(rng.choice(["I'm not able to quote pricing over the phone, but the technician will go over pricing with you on site.",
                          "Pricing depends on what the technician finds, so they'll provide a quote when they arrive."]))
            if lost_here:
                C(rng.choice(["I'd really like a number before someone comes out. I'll call around.",
                              "I'm not booking without knowing the price. Never mind."]))
            else:
                C(rng.choice(["Okay, I guess that's fine.", "Alright."]))
        elif act == "ADDRESS_LOOP":
            A("Can I get the service address?")
            C(f"It's {call['addr']}.")
            A(rng.choice(["Sorry, could you repeat the street name for me?", "Apologies, could you spell the street name?"]))
            C(f"{call['addr'].split(' ', 1)[1]}." + rng.choice([" Like I said.", "", " Can you hear me?"]))
            A(rng.choice(["Sorry, could you repeat the address one more time?", "Apologies, could you spell that address again?"]))
            if lost_here:
                C(rng.choice(["This is taking forever. Forget it.", "I've said it three times. I'll just call back later."]))
            else:
                C(f"{call['addr']}. Got it this time?")
        elif act == "INFO_OK":
            A("Can I get your name and the service address?")
            C(f"{name}, {call['addr']}.")
            A(rng.choice(["Thank you. I have that.", "Perfect, thanks."]))
        elif act == "ALT_OFFERED":
            C(f"Can you do {call['want_day']} {call['want_window']}?")
            d2, d3 = rng.sample([d for d in WEEKDAYS if d != call['want_day']], 2)
            A(f"{call['want_day']} {call['want_window']} is full, but the closest openings I have are {d2} {rng.choice(WINDOWS)} or {d3} {rng.choice(WINDOWS)}. I can also add you to our cancellation waitlist. Would either of those work?")
            if lost_here:
                C("Neither of those works for me, sorry.")
            else:
                C(f"{d2} works.")
        elif act == "NO_ALT_SLOT":
            C(f"Can you do {call['want_day']} {call['want_window']}?")
            A(rng.choice([f"I'm sorry, we don't have any openings {call['want_day']} {call['want_window']}.",
                          f"Unfortunately we don't have that window available on {call['want_day']}."]))
            if lost_here:
                C(rng.choice(["Oh. Okay, I'll call back another time.", "Never mind then, I'll look elsewhere."]))
            elif lingering:
                C("Oh.")
            else:
                C("What do you have after that?")
                A(f"The next opening is {rng.choice(WEEKDAYS)} {rng.choice(WINDOWS)}.")
                C("Fine, put me down for that.")
        elif act == "SLOT_OK":
            C(f"Can you do {call['want_day']} {call['want_window']}?")
            A(f"Yes, I have {call['want_day']} {call['want_window']} available.")
        elif act == "BOOKING_ERROR":
            A(rng.choice(["I'm having trouble accessing our schedule right now. I'll have someone from the office call you back to confirm.",
                          "I'm sorry, I'm having trouble with our scheduling system at the moment. Someone will call you back shortly."]))
            if lost_here:
                C(rng.choice(["Okay, whatever. Bye.", "I'll just call another company."]))
            else:
                C("Okay, please do.")
        elif act == "SHOPPER_CLOSE":
            A("Great, would you like me to lock that in for you?")
            if lost_here:
                C(rng.choice(["Let me check with a couple of other companies first and I'll call back.",
                              "Not yet, I'm still comparing quotes. Thanks though."]))
            else:
                C("Yes, let's book it.")
        elif act == "CONFIRM":
            A(rng.choice(["You're all set. You'll get a text confirmation with the technician's name before they arrive.",
                          "Great, you're booked. We'll text you a confirmation shortly."]))
            C(rng.choice(["Thank you!", "Great, thanks so much.", "Perfect, thank you."]))
        if lost_here or lingering:
            for i in range(len(T) - 1, n_before - 1, -1):
                if T[i][0] == "agent":
                    T[i] = ["agent", T[i][1], "FATAL"]
                    break
        risky = act in ("ROUTINE_FOR_URGENT", "PRICE_DEFLECT", "PRICE_QUOTE", "NO_ALT_SLOT", "EMERGENCY_ROUTE")
        if (lingering or (risky and not lost_here and rng.random() < 0.3)) and T[-1][0] == "caller":
            T[-1] = ["caller", rng.choice(["Hmm. Okay...", "I guess. Okay.", "Uh, okay.", "Hm, alright."])]
    if not booked and not any(f is True for _, _, f in events):
        C(rng.choice(["Actually, you know what, let me talk to my husband and call you back.",
                      "Actually, I think I'm going to hold off for now. Thanks.",
                      "You know, let me think about it and I'll call back.",
                      "Sorry, something came up. I'll call back later."]))
    if not booked:
        A(rng.choice(["I understand. Feel free to call us back anytime.", "No problem, have a good day.",
                      "Understood. We're here if you need us."]))
    merged, fatal_idx = [], None
    for item in T:
        spk, text = item[0], item[1]
        if merged and merged[-1][0] == spk:
            merged[-1][1] = merged[-1][1] + " " + text
        else:
            merged.append([spk, text])
        if len(item) > 2:
            fatal_idx = len(merged) - 1
    return merged, fatal_idx


def make_call(rng, cust, cid, day):
    svc_list = SERVICES[cust["trade"]]
    weights = [3, 2, 1.2, 0.8]
    sidx = rng.choices(range(len(svc_list)), weights=weights)[0]
    label, value, p_emerg, probs, urg = svc_list[sidx]
    emergency = rng.random() < p_emerg and bool(urg)
    traits = {"price": rng.betavariate(2, 3), "urgency": rng.betavariate(3, 2) if emergency else 0.0,
              "flex": rng.betavariate(2.5, 2), "patience": rng.betavariate(3, 2)}
    call = {
        "id": f"{cust['id'][:3].upper()}-{cid:05d}",
        "cust": cust["id"], "day": day, "service": label, "value": value,
        "emergency": emergency,
        "problem": rng.choice(urg if emergency else probs),
        "mild_cue": (not emergency) and p_emerg > 0 and rng.random() < 0.18,
        "asks_price": rng.random() < 0.15 + 0.55 * traits["price"],
        "addr_loop": rng.random() < 0.06,
        "slot_full": rng.random() < cust["tight"],
        "tool_error": rng.random() < 0.025,
        "shopping": rng.random() < 0.06,
        "traits": traits,
        "name": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
        "addr": f"{rng.randint(100, 9899)} {rng.choice(STREETS)}",
        "want_day": rng.choice(WEEKDAYS), "want_window": rng.choice(WINDOWS),
        "render_seed": rng.randrange(1 << 30),
        "u": {k: rng.random() for k in ["recognise", "false_alarm", "quote", "alt", "leave_triage", "leave_price",
                                         "leave_info", "leave_slot", "leave_book", "leave_close", "leave_quote", "linger"]},
        "judge_noise": rng.gauss(0, 1),
    }
    return call


def judge_score(p_exante, replay_booked, changed, act_changed, noise):
    """Simulated LLM judge reading a replayed transcript. It sees the caller's
    situation (p_exante) and the simulated caller's reaction (a noisy signal of
    the outcome), and it is optimistic about replays where the 'fixed'
    behaviour appears, a known LLM-as-judge failure mode."""
    bias = 0.0
    if changed:
        bias = {"PRICE_QUOTE": 1.0, "ALT_OFFERED": 0.7, "EMERGENCY_ROUTE": 0.9}.get(act_changed, 0.5)
    signal = 1.1 if replay_booked else -1.1
    return logistic(logit(0.05 + 0.9 * p_exante) + signal + bias + 0.9 * noise)


def exante(call, cust, policy):
    """P(book) given caller traits and the acts this policy picks (no leave draws)."""
    saved = dict(call["u"])
    for k in saved:
        if k.startswith("leave_"):
            call["u"][k] = 2.0  # never below a hazard
    _, _, p, _ = simulate(call, cust, policy)
    call["u"] = saved
    return p


def main():
    rng = random.Random(SEED)
    calls_out, cf = [], {c["id"]: [] for c in CANDIDATES}
    replays = {c["id"]: {} for c in CANDIDATES}
    cust_by_id = {c["id"]: c for c in CUSTOMERS}
    n = 0
    for day in range(1, DAYS + 1):
        for cust in CUSTOMERS:
            k = max(0, int(round(rng.gauss(cust["leads_per_day"], cust["leads_per_day"] ** 0.5))))
            for _ in range(k):
                n += 1
                call = make_call(rng, cust, n, day)
                booked, ev, p_book, mode = simulate(call, cust, "baseline")
                turns, truth_turn = render(call, cust, ev, booked, mode, random.Random(call["render_seed"]))
                lingered = any(f == "linger" for _, _, f in ev)
                dur = sum(len(t[1]) for t in turns) // 14 + 20
                calls_out.append({
                    "id": call["id"], "cust": call["cust"], "day": day, "svc": call["service"],
                    "value": call["value"], "booked": booked, "dur": dur, "turns": turns,
                    "truth": {"mode": mode, "turn": truth_turn, "lingered": lingered},
                })
                for cand in CANDIDATES:
                    b2, ev2, p2, m2 = simulate(call, cust, cand["id"])
                    acts1 = [a for _, a, _f in ev]
                    acts2 = [a for _, a, _f in ev2]
                    changed = acts1 != acts2 or b2 != booked
                    act_changed = None
                    for a, b in zip(acts1 + [None] * 8, acts2):
                        if a != b:
                            act_changed = b
                            break
                    f = judge_score(exante(call, cust, cand["id"]), b2, changed, act_changed, call["judge_noise"])
                    cf[cand["id"]].append([1 if b2 else 0, round(f, 4), 1 if changed else 0])
                    if changed:
                        t2, _ft = render(call, cust, ev2, b2, m2, random.Random(call["render_seed"]))
                        # store the replay from the first diverging turn
                        k0 = 0
                        while k0 < min(len(turns), len(t2)) and turns[k0] == t2[k0]:
                            k0 += 1
                        replays[cand["id"]][call["id"]] = {"from": k0, "turns": t2[k0:]}
    meta = {
        "generated_by": "data/generate.py", "seed": SEED, "period": "2026-10-01 to 2026-10-30",
        "customers": [{k: c[k] for k in ["id", "name", "trade", "city", "dispatch_fee", "emergency_fee"]}
                      for c in CUSTOMERS],
        "note": "Synthetic. Fictional businesses, callers and addresses.",
    }
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "calls.json"), "w") as fh:
        json.dump({"meta": meta, "calls": calls_out}, fh, separators=(",", ":"))
    with open(os.path.join(here, "counterfactuals.json"), "w") as fh:
        json.dump({"candidates": CANDIDATES, "outcomes": cf, "replays": replays}, fh, separators=(",", ":"))
    booked = sum(c["booked"] for c in calls_out)
    print(f"leads={len(calls_out)} booked={booked} rate={booked / len(calls_out):.3f}")
    for cand in CANDIDATES:
        ys = [r[0] for r in cf[cand["id"]]]
        print(cand["id"], "true rate", round(sum(ys) / len(ys), 4), "changed", sum(r[2] for r in cf[cand["id"]]))


if __name__ == "__main__":
    main()
