import re


POSITIVE_TERMS = [
    "अच्छा",
    "अच्छी",
    "अच्छे",
    "बढ़िया",
    "बहुत अच्छा",
    "शानदार",
    "उत्तम",
    "मस्त",
    "पसंद",
    "सही",
    "खुश",
    "बेहतरीन",
    "आसान",
    "पैसे वसूल",
    "समय पर",
    "उपयोगी",
    "बढ़िया अनुभव",
    "अच्छा अनुभव",
    "जरूर",
    "खरीदूंगा",
    "खरीदूंगी",
    "લાજવાબ",
    "સરસ",
    "સારું",
    "સારી",
    "સારા",
    "સારો",
    "ઉત્તમ",
    "મસ્ત",
    "ગમ્યું",
    "ખુશ",
    "સરળ",
    "પૈસા વસૂલ",
    "સમયસર",
    "ઉપયોગી",
    "સારો અનુભવ",
    "ખરીદીશ",
    "acha",
    "achha",
    "accha",
    "badhiya",
    "shandar",
    "mast",
    "saras",
    "saru",
    "saro",
    "majama",
]

NEGATIVE_TERMS = [
    "खराब",
    "बेकार",
    "बुरा",
    "बुरी",
    "घटिया",
    "नापसंद",
    "टूटा",
    "धीमा",
    "महंगा",
    "समस्या",
    "बर्बादी",
    "लेट",
    "हैंग",
    "संतुष्ट नहीं",
    "अनुसार नहीं",
    "બેકાર",
    "નકામું",
    "ખરાબ",
    "નબળું",
    "તૂટેલું",
    "ધીમું",
    "મોંઘું",
    "સમસ્યા",
    "બગાડ",
    "મોડે",
    "બગડી",
    "બગડ્યો",
    "હેંગ",
    "સંતોષ નથી",
    "મુજબ નથી",
    "kharab",
    "bekar",
    "bakwas",
    "ghatiya",
    "bura",
    "tuta",
    "dhimu",
    "monghu",
]

STRONG_NEGATIVE_TERMS = [
    "पसंद नहीं",
    "अच्छा नहीं",
    "अच्छी नहीं",
    "सही नहीं",
    "संतुष्ट नहीं",
    "अनुसार नहीं",
    "नहीं है",
    "નથી ગમ્યું",
    "સારું નથી",
    "સારી નથી",
    "ગમ્યું નથી",
    "સંતોષ નથી",
    "મુજબ નથી",
    "pasand nahi",
    "acha nahi",
    "achha nahi",
    "accha nahi",
    "saru nathi",
    "saro nathi",
    "gamyu nathi",
]


def multilingual_sentiment_tokens(text):
    lower_text = str(text).lower()
    if any(term in lower_text for term in STRONG_NEGATIVE_TERMS):
        return " poor quality bad experience defective item not worth the price"

    positive_hits = sum(term in lower_text for term in POSITIVE_TERMS)
    negative_hits = sum(term in lower_text for term in NEGATIVE_TERMS)

    if positive_hits > negative_hits:
        return " great quality excellent build highly recommended works perfectly"
    if negative_hits > positive_hits:
        return " poor quality bad experience defective item not worth the price"
    return ""


def clean_text(text):
    extra_tokens = multilingual_sentiment_tokens(text)
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"(.)\1+", r"\1\1", text)
    return f"{text}{extra_tokens}"
