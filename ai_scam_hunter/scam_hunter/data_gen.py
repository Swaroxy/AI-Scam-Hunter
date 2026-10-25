"""Synthetic (fully fictional) email/SMS generator.

Why synthetic? It keeps the project license-clean and lets us control dataset-quality
lessons: every message belongs to a template `group` (its body), so we can split by group
and prevent template leakage between train and test. We also inject duplicates on purpose
so the deduplication step has something real to remove.

All brands, links and phone numbers are fictional (.example domains, 555 numbers).
"""
from __future__ import annotations

import json
import random

import pandas as pd

from .config import RAW_CSV, SEED

SIGN_TYPES = [
    "urgency", "credential_request", "suspicious_link", "payment_request", "prize_lure",
    "threat", "impersonation", "secrecy", "remote_access", "too_good_to_be_true",
]

NAMES = ["Alex", "Sam", "Priya", "Jordan", "Mina", "Carlos", "Dana", "Omar", "Lena", "Ravi"]
BANKS = ["Northwind Bank", "Contoso Credit Union", "Fabrikam Pay", "Litware Wallet"]
SHOPS = ["Fabrikam Shop", "Woodgrove Market", "Proseware Store"]
BAD_WORDS = ["verify", "secure", "login", "claim", "update", "pay"]
GOOD_DOMAINS = ["help.northwind-bank.example", "www.fabrikam-shop.example", "support.woodgrove.example"]
PRIZES = ["a brand new smartphone", "a $1,000 shopping voucher", "a luxury holiday package", "a gaming laptop"]

CATEGORIES: dict[str, dict] = {
    # ------------------------------------------------------------------ suspicious
    "bank_phishing": dict(
        label=1, n=70, signs=["impersonation", "urgency", "credential_request", "suspicious_link"],
        subjects=["Urgent: account suspended", "Security alert", "Verify your account immediately", "Your card has been locked"],
        openers=["Dear customer,", "Security alert from {bank}.", "Important notice:", "Hello {name},"],
        bodies=[
            "We detected unusual sign-in activity on your {bank} account. Your account will be suspended in {hours} hours unless you verify your identity.",
            "Your {bank} online banking access has been limited after a failed security check. Confirm your username and password to restore access.",
            "A payment of {amount} to an unknown recipient is pending on your {bank} card. If this was not you, log in now to cancel it.",
            "Your {bank} debit card has been temporarily locked. Re-enter your card number and PIN to unlock it.",
            "Action required: your {bank} profile is out of date and online access will be disabled. Update your details to avoid closure.",
            "We could not complete your recent transfer. Please reply with your online banking details so we can retry it.",
            "Our records show your {bank} security answers have expired. Reply with the new answers to keep your profile active.",
        ],
        ctas=["Verify now: {bad_link}", "Log in here: {bad_link}", "Secure your account: {bad_link}"],
    ),
    "prize_lottery": dict(
        label=1, n=70, signs=["prize_lure", "urgency", "payment_request", "suspicious_link"],
        subjects=["You have won!", "Claim your prize", "Winner notification", "Your reward is waiting"],
        openers=["Congratulations {name}!", "You're a winner!", "Great news!"],
        bodies=[
            "You have been randomly selected to win {prize}. Claim your reward within {hours} hours or it will be given to someone else.",
            "You are the lucky winner of our monthly draw worth {amount}! A small processing fee is required before we can release your funds.",
            "Your mobile number won the {bank} anniversary lottery. Reply with your full name and address to receive {prize}.",
            "Exclusive offer: you qualify for {prize} for free. Just cover the {fee} shipping charge to claim it today.",
            "Final notice: an unclaimed prize of {amount} is waiting in your name. Confirm your details to collect it.",
            "Thanks for completing our survey. As a thank-you you can pick a free gift, you only pay shipping.",
            "Your name was drawn in our community raffle. Send a copy of your ID to receive the prize.",
        ],
        ctas=["Claim here: {bad_link}", "Tap to claim: {bad_link}", "Reply YES and visit {bad_link}"],
    ),
    "delivery_fee": dict(
        label=1, n=70, signs=["impersonation", "payment_request", "urgency", "suspicious_link"],
        subjects=["Delivery failed", "Parcel on hold", "Customs fee required", "Redelivery notice"],
        openers=["Tailspin Couriers:", "Notice from Tailspin Couriers:", "Hello {name},"],
        bodies=[
            "Your parcel {tracking} could not be delivered because of an unpaid customs fee of {fee}. Pay the fee to schedule redelivery.",
            "We tried to deliver your package today but the address was incomplete. Update your address and pay a {fee} redelivery charge or it will be returned.",
            "Package {tracking} is held at our depot. A {fee} handling fee must be paid within {hours} hours.",
            "Your order from {shop} is delayed. Confirm your card details to cover the {fee} shipping adjustment.",
            "Delivery attempt failed. Your parcel will be destroyed in {days} days unless you reschedule and pay the small fee.",
            "Your package needs a signature. Confirm your address and make a small handling payment to release it.",
            "Customs needs an extra payment before your order from {shop} can be released.",
        ],
        ctas=["Pay here: {bad_link}", "Reschedule: {bad_link}", "Track and pay: {bad_link}"],
    ),
    "tech_support": dict(
        label=1, n=70, signs=["impersonation", "urgency", "remote_access", "payment_request", "threat"],
        subjects=["Virus detected", "Your subscription was renewed", "Device compromised", "Storage full - files will be deleted"],
        openers=["Litware Security Alert:", "Attention user,", "Hello {name},"],
        bodies=[
            "Our system has detected a virus on your computer. Call our certified support team immediately to remove it before your data is lost.",
            "Your Litware Antivirus subscription was renewed for {amount}. If you did not authorise this, call us now to get a refund.",
            "Your device has been compromised and hackers may access your photos. Install our remote tool so a technician can fix it.",
            "Your Litware cloud storage is full and files will be deleted tomorrow. Call support to upgrade and keep your data safe.",
            "We found a suspicious login from another country. Allow our agent to connect to your computer so we can secure your account.",
            "Your router was flagged for unusual traffic. Our technician needs to log in to your computer to fix it.",
            "Your Litware licence has expired. Pay the renewal fee by phone to stay protected.",
        ],
        ctas=["Call {phone} now.", "Call toll-free {phone} immediately.", "Download the support tool: {bad_link}"],
    ),
    "gift_card_boss": dict(
        label=1, n=70, signs=["impersonation", "payment_request", "urgency", "secrecy"],
        subjects=["Quick favour", "Urgent request", "Are you at your desk?", "Confidential task"],
        openers=["Hi {name},", "{name}, quick one:", "Hello,"],
        bodies=[
            "I'm in a meeting and can't talk. I need you to buy {n} gift cards of {amount} each for a client. Keep this between us and send me the codes.",
            "Are you available? I need a quick favour. Please purchase gift cards today and send photos of the codes, I will reimburse you.",
            "Please process an urgent wire transfer of {amount} to a new vendor. Do not discuss this with anyone until it is done.",
            "I need you to handle something confidential. Buy prepaid cards and send me the numbers as soon as you can.",
            "Change of plans: please update the payroll bank details to the new account I will send and confirm back quietly.",
            "Are you free for a quick task? Send me your number and I will explain, it is a bit sensitive.",
            "Please pay the attached invoice today to the updated bank account. The old account is closed.",
        ],
        ctas=["Reply as soon as you see this.", "Text me back, I'm not on calls.", "Let me know once it's done."],
    ),
    "investment_advance_fee": dict(
        label=1, n=70, signs=["too_good_to_be_true", "payment_request", "urgency"],
        subjects=["Guaranteed returns", "Unclaimed funds", "Private investment group", "Exclusive opportunity"],
        openers=["Hello friend,", "Dear {name},", "Opportunity alert:"],
        bodies=[
            "Our trading bot guarantees {pct}% returns every week. Deposit {amount} in crypto today and watch it grow with zero risk.",
            "A wealthy donor wants to transfer {amount} to you. We only need a small advance fee to release the funds.",
            "Limited spots are left in our private investment group. Members doubled their money in {days} days. Send the deposit now to join.",
            "I write on behalf of a bank director with an unclaimed inheritance. You are the named beneficiary, so contact us to claim your share.",
            "I found your profile and want to share a secret investment tip that made me rich. Only the first 20 people get in.",
            "Hi {name}, is this still your number? A friend gave it to me. I also trade crypto, want me to show you how?",
            "I work with a small team that helps people grow savings quickly. Send {amount} to start and I will double it.",
        ],
        ctas=["Join here: {bad_link}", "Reply today to secure your place.", "Send your details to {bad_email}"],
    ),
    # ------------------------------------------------------------------ legitimate
    "meeting_work": dict(
        label=0, n=130, signs=[],
        subjects=["Project sync", "Reschedule one-on-one", "Standup notes", "Report draft ready"],
        openers=["Hi {name},", "Hello team,", "Hi all,"],
        bodies=[
            "Reminder: the project sync is scheduled for {date} at {time}. The agenda is attached, please add any topics before the meeting.",
            "Can we move our one-on-one to {time} tomorrow? I have a clash with the planning session.",
            "Here are the notes from today's standup. Action items are at the bottom, please update your tickets by Friday.",
            "The quarterly report draft is ready for review. Please leave comments in the shared document by {date}.",
            "Thanks for sending the slides. I made a few small edits and left suggestions for the intro.",
            "Please review and approve the vendor invoice for {amount} by {date}. It is attached in the finance system.",
            "Urgent: the server maintenance window starts at {time} tonight. Please save your work and log out before then.",
        ],
        ctas=["Thanks!", "See you then.", "Let me know if that works.", "Best, {name2}"],
    ),
    "receipt_order": dict(
        label=0, n=130, signs=[],
        subjects=["Your receipt", "Order shipped", "Subscription renewal", "Refund processed"],
        openers=["Hello {name},", "Thanks for your purchase!", "Order update:"],
        bodies=[
            "Thank you for your order from {shop}. Your receipt for {amount} is attached and your items will ship within {days} business days.",
            "Your {shop} order has shipped. You can track it any time from the Orders page after signing in on our official website.",
            "Your subscription renews on {date} for {amount}. You can manage or cancel it any time in your account settings.",
            "We received your return. A refund of {amount} will appear on your original payment method within {days} business days.",
            "Your invoice for {amount} is ready. Payment is due by {date}; no action is needed if you have autopay enabled.",
            "Your payment of {amount} is due. Pay securely at {good_link} or by bank transfer using the details on your invoice.",
            "Verify your email address to finish setting up your {shop} account: {good_link}",
        ],
        ctas=["Thanks for shopping with us.", "Questions? Visit {good_link}", "Contact support at {good_link}"],
    ),
    "appointment": dict(
        label=0, n=130, signs=[],
        subjects=["Appointment reminder", "Library due date", "Table booking", "Service booked"],
        openers=["Hello {name},", "Friendly reminder:", "Hi {name},"],
        bodies=[
            "This is a reminder of your appointment on {date} at {time}. Reply C to confirm or call the clinic to reschedule.",
            "Your library book is due on {date}. You can renew it online or at the front desk.",
            "Your table for {n} is booked for {date} at {time}. We look forward to seeing you.",
            "The parent-teacher meeting is on {date} at {time} in the main hall. Please bring your child's folder.",
            "Your car service is booked for {date}. Please drop off the keys by {time}.",
            "Your appointment fee of {fee} will be taken at the front desk. Please arrive 10 minutes early.",
            "Immediate action needed: the school bus route changes on {date}. Please check the new times.",
        ],
        ctas=["See you soon.", "Reply STOP to opt out.", "Thank you!"],
    ),
    "security_notice_safe": dict(  # hard negatives: sound security-ish but are benign
        label=0, n=130, signs=[],
        subjects=["Your verification code", "New sign-in noticed", "Statement ready", "Password changed", "Payment approved"],
        openers=["{bank} notice:", "Hello {name},", "Automated message:"],
        bodies=[
            "Your verification code is {code}. It expires in 10 minutes. Never share this code with anyone, including our staff.",
            "We noticed a new sign-in to your {bank} account from a new device. If this was you, no action is needed. If not, change your password in the official app.",
            "Your {bank} statement for this month is ready. View it in the official app. We will never ask for your password by message.",
            "Your password was changed successfully. If you did not make this change, call {bank} using the number on the back of your card.",
            "A payment of {amount} to {shop} was approved on your {bank} card. No action is needed.",
            "Your {bank} card was locked after three wrong PIN attempts. Unlock it in the official app or at any branch. We never ask for your PIN.",
            "To protect your account we need you to confirm a recent purchase. Open the official app and review your activity. Never share your password.",
        ],
        ctas=["Do not reply to this message.", "Visit {good_link} for help.", "Thank you for banking with us."],
    ),
    "social_newsletter": dict(
        label=0, n=130, signs=[],
        subjects=["Dinner this week?", "Happy birthday!", "Weekly newsletter", "Neighbourhood clean-up"],
        openers=["Hey {name}!", "Hi {name},", "Hello neighbours,"],
        bodies=[
            "Are we still on for dinner on {date}? I found a new place near the station.",
            "Hope you have a wonderful birthday, let's catch up this weekend.",
            "This week's newsletter: three tips for better sleep, a lentil soup recipe, and our community events calendar.",
            "Don't forget the neighbourhood clean-up on {date}. Gloves and bags are provided, bring a friend!",
            "Mum asked if you can pick up some bread on your way home. Also, the plumber is coming at {time}.",
            "Claim your free ticket to the community fair! Show this message at the gate on {date}.",
            "Congratulations on the new job! Dinner is on me, I will pay.",
        ],
        ctas=["Talk soon!", "Unsubscribe any time using the link at the bottom.", "Cheers!"],
    ),
}


def _slots(rng: random.Random) -> dict:
    slug = rng.choice(["northwind", "contoso", "fabrikam", "litware", "tailspin"])
    return dict(
        name=rng.choice(NAMES), name2=rng.choice(NAMES), bank=rng.choice(BANKS), shop=rng.choice(SHOPS),
        amount=f"${rng.choice([19, 49, 89, 250, 499, 1200, 4800])}.{rng.choice(['00', '50', '99'])}",
        fee=f"${rng.choice([1, 2, 3, 5, 9])}.{rng.choice(['49', '95', '99'])}",
        hours=rng.choice([6, 12, 24, 48]), days=rng.choice([2, 3, 5, 7]), n=rng.choice([2, 3, 4, 5]),
        pct=rng.choice([40, 80, 150, 300]), code=rng.randint(100000, 999999),
        tracking=f"TC{rng.randint(10**8, 10**9 - 1)}", prize=rng.choice(PRIZES),
        phone=f"1-800-555-{rng.randint(1000, 9999)}",
        date=rng.choice(["Mon 3 Nov", "Tue 4 Nov", "Thu 6 Nov", "Fri 14 Nov", "Mon 17 Nov"]),
        time=rng.choice(["9:00", "10:30", "13:00", "15:45", "17:15"]),
        bad_link=rng.choice([
            f"http://{slug}-{rng.choice(BAD_WORDS)}-{rng.choice(BAD_WORDS)}.example/{rng.randint(100, 999)}",
            f"https://{slug}-{rng.choice(BAD_WORDS)}.example/{rng.randint(100, 999)}",
            f"https://short.example/{rng.choice('abcdefgh')}{rng.randint(1000, 9999)}",
        ]),
        bad_email=f"{rng.choice(['claims', 'agent', 'desk'])}{rng.randint(10, 99)}@{slug}-{rng.choice(BAD_WORDS)}.example",
        good_link=f"https://{rng.choice(GOOD_DOMAINS)}/help",
    )


# Shared by BOTH classes so openers/closers/subjects alone cannot reveal the label.
GENERIC_OPENERS = ["Hi {name},", "Hello,", "Dear {name},", "Good morning,", "Hi,"]
GENERIC_CTAS = ["Thanks.", "Please reply when you can.", "Let me know.", "Thank you for your time.", "Best regards.", "Regards, {name2}"]
GENERIC_SUBJECTS = ["Hello", "Quick question", "Important", "Reminder", "Update", "Re: your request", "Action needed"]
P_GENERIC = 0.6

_LEET = {"verify": "v3rify", "account": "acc0unt", "password": "passw0rd", "claim": "cl@im", "urgent": "urg3nt",
         "payment": "paym3nt", "gift": "g1ft", "bank": "b4nk", "free": "fr33"}


def obfuscate(text: str, rng: random.Random) -> str:
    """Scammers dodge keyword filters with leetspeak; this makes word-level baselines work harder."""
    hits = [w for w in _LEET if w in text.lower()]
    for w in rng.sample(hits, k=min(2, len(hits))):
        text = text.replace(w, _LEET[w]).replace(w.capitalize(), _LEET[w].capitalize())
    return text


def generate_dataset(seed: int = SEED, dup_exact: float = 0.06, dup_near: float = 0.03) -> pd.DataFrame:
    """Return a shuffled DataFrame of fictional messages, including some injected duplicates."""
    rng = random.Random(seed)
    rows = []
    for cat, spec in CATEGORIES.items():
        for _ in range(spec["n"]):
            body_idx = rng.randrange(len(spec["bodies"]))
            s = _slots(rng)
            body = spec["bodies"][body_idx].format(**s)
            cta = rng.choice(spec["ctas"])
            if "{bad_link}" in cta and rng.random() < 0.35:  # many real scams carry no link
                cta = rng.choice(["Reply to this message.", "Reply YES to continue.", "Respond today."])
            elif rng.random() < P_GENERIC:
                cta = rng.choice(GENERIC_CTAS)
            cta = cta.format(**s)
            opener = rng.choice(GENERIC_OPENERS if rng.random() < P_GENERIC else spec["openers"]).format(**s)
            if rng.random() < 0.55:
                channel = "email"
                subject = rng.choice(GENERIC_SUBJECTS if rng.random() < P_GENERIC else spec["subjects"])
                text = f"Subject: {subject}\n\n{opener} {body} {cta}".strip()
            else:
                channel = "sms"
                text = (f"{opener} " if rng.random() < 0.4 else "") + f"{body} {cta}"
                text = text.strip()
            if spec["label"] == 1 and rng.random() < 0.2:
                text = obfuscate(text, rng)
            rows.append(dict(channel=channel, text=text, label=spec["label"], category=cat,
                             group=f"{cat}:{body_idx}", signs=json.dumps(spec["signs"])))
    df = pd.DataFrame(rows)

    # Inject duplicates on purpose so the quality step has real work to do.
    exact = df.sample(frac=dup_exact, random_state=seed)
    near = df.sample(frac=dup_near, random_state=seed + 1).copy()
    near["text"] = near["text"].str.upper().str.replace(" ", "  ", n=1, regex=False) + " !!"
    df = pd.concat([df, exact, near], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "id", range(len(df)))
    return df


if __name__ == "__main__":
    data = generate_dataset()
    data.to_csv(RAW_CSV, index=False)
    print(f"Wrote {len(data)} rows to {RAW_CSV}")
    print(data["label"].value_counts().rename({0: "legitimate", 1: "suspicious"}))
