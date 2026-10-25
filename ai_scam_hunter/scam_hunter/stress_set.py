"""Hand-written, out-of-template stress test (fictional). Never used for training or tuning.

Template-generated test data is easy because the wording is repetitive. This small set uses
novel phrasing, typos/leetspeak, and benign messages containing scammy vocabulary, so it
exposes models that memorised keywords rather than learned the pattern.
"""
import pandas as pd

_SUSPICIOUS = [
    "Hey its me, got a new number. Can u grab a few iTunes cards for my cousin? ill pay u back tonight, dont tell anyone yet",
    "Litware Wallet: your acc0unt is on hold. v3rify your identity today at http://litware-v3rify.example/k2 to avoid permanent closure",
    "Your TailSpin parcel is stuck. Pay $1.99 to release it: https://short.example/p7731",
    "Hi, this is Dana from IT. Your mailbox is over quota. Send me your password and I will fix it from here.",
    "Ur phone number was picked in our weekly giveaway!! Text back your home address to get ur free tablet",
    "I have a guaranteed way to turn $500 into $5000 in a week. No risk. DM me for the wallet address.",
    "Northwind Bank fraud team: we blocked a transfer. To reverse it, read us the 6 digit code we just texted you.",
    "Final reminder: your tax refund of $842.10 is pending. Submit your bank login to receive it https://refund-claim.example/x",
    "Boss here. I am boarding a flight and cannot call. Wire $3,200 to the vendor below today, I will explain later.",
    "Congrats!! U have been chosen for a cash reward. Pay the $4.99 admin fee to unlock ur money",
    "Your computer is sending errors to hackers. Do not turn it off. Call 1-800-555-0142 for a free fix.",
    "Hello dear, I am a widow with $2.5M I wish to donate to charity. Kindly send your bank details and a small fee for the lawyer.",
    "Contoso Credit Union: unusual login. Your card will be disabled in 2 hrs. Confirm card number and PIN: http://contoso-secure.example/c",
    "We could not deliver your item. Click to re-enter your card for the redelivery fee. Offer expires today.",
    "Quick favour please, buy 4 Google Play cards at $100 and send me pics of the backs. Dont call, in a meeting.",
    "Earn $300 a day from home, just send a $50 starter kit fee. Limited seats, answer within the hour.",
    "Security notice: install this app so our agent can remotely remove the virus: https://support-fix.example/get",
    "Your account will be deleted unless you verify your password here: https://fabrikam-login.example/v",
    "Hi love, I know we just met online but I am stuck abroad and need $600 for a ticket. I will repay you, promise.",
    "Fabrikam Pay: a purchase of $1,299 was made on your account. If this wasn't you, reply with your password to cancel it.",
]
_LEGIT = [
    "Hey, running 10 mins late. Start without me and I'll pick up the coffees on the way.",
    "Your Northwind Bank verification code is 482913. It expires in 10 minutes. Never share it with anyone.",
    "Reminder: your dentist appointment is tomorrow at 9:30. Reply C to confirm. Please arrive early to fill in the forms.",
    "Urgent: the office printer is out of toner and the board pack is due at 3pm. Can someone grab a spare from stores?",
    "Your Fabrikam Shop order #7719 has shipped. Track it by signing in at https://www.fabrikam-shop.example/orders",
    "Congratulations on your promotion! Drinks tonight, my treat.",
    "Your invoice for $240.00 is due on Friday. Pay at https://help.northwind-bank.example/invoices or by bank transfer.",
    "We're sorry your parcel was late. A $5 credit has been added to your account, no action needed.",
    "Can you send me the signed contract by end of day? Legal needs it before the call tomorrow.",
    "Password reset requested. If this was you, use the official app to choose a new one. If not, ignore this message.",
    "Your card ending 4417 was used for $18.20 at Woodgrove Market. Not you? Call the number on the back of your card.",
    "Free workshop on budgeting this Saturday at the community centre. Everyone welcome, no registration needed.",
    "Dad's birthday is on the 14th. Let's all chip in for the gift, send me what you can when you get a chance.",
    "Your subscription will renew on 3 Nov for $9.99. Manage it any time in settings.",
    "Hi Sam, the landlord says the rent is due on the 1st as usual. Same account details as before.",
    "Important: school is closed tomorrow due to the storm. Classes resume Thursday.",
    "Your refund of $32.00 has been issued and should appear in 3 to 5 business days.",
    "Final call for the team lunch order! Reply with your choice by noon so I can place it.",
    "Winner of this month's staff award: Priya from Accounts. Join us at 4pm to celebrate.",
    "Your Litware account storage is 90% full. You can free up space or upgrade from the settings page whenever you like.",
]


def load_stress_set() -> pd.DataFrame:
    rows = [{"text": t, "label": 1, "category": "stress_suspicious"} for t in _SUSPICIOUS]
    rows += [{"text": t, "label": 0, "category": "stress_legitimate"} for t in _LEGIT]
    return pd.DataFrame(rows)
