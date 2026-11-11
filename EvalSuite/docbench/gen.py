#!/usr/bin/env python3
"""docbench v1 generator: synthetic OCR-style text documents with exact ground truth, deterministic (fixed seed).
Five types x 30: invoices, receipts, forms, bank statements (tables), contracts / letters; in each type 24 English
(US / UK / Ireland) and 2 each German, French, Bulgarian (20 %). Hard in the IDP way: distractor numbers and dates
(order / delivery / due dates, subtotal / total / previous balance, loyalty-card numbers, pending items), mixed date
formats with locale hints, column layouts and misaligned tables, light OCR noise in non-key text, absent fields
(null). Every gold value is printed in the text (checked by `evidence`); an EN numeric date that is ambiguous between
DD/MM and MM/DD always has a disambiguating date in the same format or a format note in the document.
Usage: python gen.py [--out set/docs.jsonl] [--show TYPE:N]"""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, os, random, sys
from decimal import Decimal, ROUND_HALF_UP

SEED = 20261025
TYPES = ["invoice", "receipt", "form", "statement", "contract"]
PLAN = ["en"] * 24 + ["de", "de", "fr", "fr", "bg", "bg"]
LANG = {"US": "en", "UK": "en", "IE": "en", "DE": "de", "FR": "fr", "BG": "bg"}
CUR = {"US": "USD", "UK": "GBP", "IE": "EUR", "DE": "EUR", "FR": "EUR", "BG": "BGN"}
C = Decimal("0.01")


def D(x) -> Decimal:
    return Decimal(str(x)).quantize(C, ROUND_HALF_UP)


# ---- names, places --------------------------------------------------------------------------------------------------
EN_A = ["Northwind", "Harbor", "Blue Ridge", "Kestrel", "Atlas", "Pinecrest", "Silverline", "Redwood", "Meridian",
        "Granite", "Oakmont", "Lumen", "Vertex", "Cobalt", "Brightwater", "Summit", "Ironbridge", "Clearwater",
        "Foxglove", "Highmoor", "Larkspur", "Westbrook", "Hollins", "Ashford", "Copperfield", "Stonegate",
        "Elmstead", "Juniper", "Marlowe", "Thornbury"]
EN_B = ["Supplies", "Logistics", "Consulting", "Engineering", "Foods", "Print Works", "Systems", "Medical",
        "Electrical", "Packaging", "Analytics", "Interiors", "Freight", "Software", "Hardware", "Textiles", "Labs",
        "Builders", "Office Solutions", "Catering"]
DE_A = ["Müller", "Schröder", "Weißgerber", "Hoffmann", "Becker", "Lindner", "Krüger", "Vogel", "Brandt", "Neumann",
        "Albrecht", "Seidel"]
DE_B = ["Bürotechnik", "Logistik", "Maschinenbau", "Elektrotechnik", "Druckerei", "Haustechnik", "Software",
        "Verpackungen", "Feinkost", "Metallbau"]
FR_A = ["Dupont", "Lefèvre", "Moreau", "Girard", "Lambert", "Fontaine", "Rousseau", "Chevalier", "Garnier", "Faure",
        "Mercier", "Bonnet"]
FR_B = ["Imprimerie", "Logistique", "Bâtiment", "Informatique", "Emballages", "Conseil", "Électricité", "Traiteur",
        "Fournitures"]
BG_A = ["Петров", "Иванов", "Георгиев", "Димитров", "Николов", "Стоянов", "Христов", "Тодоров", "Попов", "Василев"]
BG_B = ["Строймаркет", "Балкан Трейд", "Логистик", "Електроникс", "Агро", "Пак", "Софт", "Медикал", "Транс", "Принт"]

FIRST = {"US": ["James", "Emily", "Michael", "Sarah", "David", "Olivia", "Daniel", "Grace", "Thomas", "Hannah",
                "Robert", "Chloe", "William", "Megan", "Joseph", "Laura"],
         "DE": ["Jürgen", "Anna", "Lukas", "Sophie", "Matthias", "Katrin", "Florian", "Lena", "Stefan", "Miriam"],
         "FR": ["Élodie", "François", "Camille", "Julien", "Hélène", "Mathieu", "Chloé", "Sébastien"]}
LAST = {"US": ["Carter", "Novak", "Whitfield", "Okafor", "Patel", "Hughes", "Brennan", "Fletcher", "Lawson",
               "Mitchell", "Doyle", "Sinclair", "Harper", "Quinn", "Reyes", "Kowalski"],
        "DE": ["Weiß", "Schröder", "Hofmann", "Krämer", "Bauer", "Zimmermann", "Köhler", "Lehmann"],
        "FR": ["Martin", "Lefèvre", "Bernard", "Dubois", "Moreau", "Laurent", "Girard", "Rousseau"]}
BG_M = ["Иван", "Николай", "Георги", "Петър", "Димитър", "Стефан", "Христо", "Александър"]
BG_F = ["Мария", "Елена", "Десислава", "Габриела", "Йорданка", "Цветелина", "Радостина", "Петя"]
BG_PAT = ["Иванов", "Петров", "Георгиев", "Димитров", "Стефанов", "Христов", "Николаев"]
BG_LAST = ["Петров", "Иванов", "Георгиев", "Димитров", "Стоянов", "Тодоров", "Попов", "Василев", "Колев", "Ангелов"]

US_CITY = [("Springfield", "IL", "62704"), ("Austin", "TX", "78701"), ("Portland", "OR", "97205"),
           ("Columbus", "OH", "43215"), ("Raleigh", "NC", "27601"), ("Denver", "CO", "80202"),
           ("Madison", "WI", "53703"), ("Tucson", "AZ", "85701"), ("Albany", "NY", "12207"),
           ("Richmond", "VA", "23219")]
UK_CITY = [("Leeds", "LS1 4AB"), ("Bristol", "BS1 5TR"), ("Manchester", "M1 2WD"), ("Glasgow", "G2 1DY"),
           ("Cardiff", "CF10 1EP"), ("Nottingham", "NG1 5FS"), ("York", "YO1 7HH"), ("Brighton", "BN1 3XE"),
           ("Oxford", "OX1 2JD"), ("Sheffield", "S1 2HE")]
IE_CITY = [("Cork", "T12 RX6N"), ("Galway", "H91 E2K4"), ("Limerick", "V94 T9PX"), ("Dublin", "D02 X285"),
           ("Waterford", "X91 K0EK")]
DE_CITY = [("10115", "Berlin"), ("80331", "München"), ("50667", "Köln"), ("20095", "Hamburg"),
           ("60311", "Frankfurt am Main"), ("70173", "Stuttgart"), ("04109", "Leipzig"), ("01067", "Dresden"),
           ("90402", "Nürnberg"), ("28195", "Bremen")]
FR_CITY = [("75002", "Paris"), ("69002", "Lyon"), ("13001", "Marseille"), ("31000", "Toulouse"),
           ("33000", "Bordeaux"), ("44000", "Nantes"), ("67000", "Strasbourg"), ("59000", "Lille"),
           ("34000", "Montpellier"), ("35000", "Rennes")]
BG_CITY = [("1000", "София"), ("4000", "Пловдив"), ("9000", "Варна"), ("8000", "Бургас"), ("7000", "Русе"),
           ("5000", "Велико Търново"), ("6000", "Стара Загора"), ("2700", "Благоевград")]
COUNTRY_PRINT = {"US": ["USA", "United States"], "UK": ["United Kingdom", "UK"], "IE": ["Ireland"],
                 "DE": ["Deutschland", "Germany"], "FR": ["France"], "BG": ["България", "Bulgaria"]}
COUNTRY_GOLD = {"US": "United States", "UK": "United Kingdom", "IE": "Ireland", "DE": "Germany", "FR": "France",
                "BG": "Bulgaria"}

MONTHS = {"en": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                 "November", "December"],
          "de": ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober",
                 "November", "Dezember"],
          "fr": ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
                 "novembre", "décembre"],
          "bg": ["януари", "февруари", "март", "април", "май", "юни", "юли", "август", "септември", "октомври",
                 "ноември", "декември"]}
DATE_STYLES = {"US": ["m/d/Y", "m/d/y", "m-d-Y", "Mon d, Y", "Month d, Y"],
               "UK": ["d/m/Y", "d/m/y", "d Month Y", "d Mon Y", "dd-Mon-Y"],
               "IE": ["d/m/Y", "d Month Y", "d Mon Y"],
               "DE": ["d.m.Y", "d.m.y", "d. Month Y"],
               "FR": ["d/m/Y", "d.m.Y", "d1 month Y"],
               "BG": ["d.m.Y г.", "d.m.Y", "d month Y г."]}
AMBIG = {"m/d/Y", "m/d/y", "m-d-Y", "d/m/Y", "d/m/y"}            # EN numeric styles: DD/MM vs MM/DD


# ---- OCR noise ------------------------------------------------------------------------------------------------------
SUBS = [("l", "1"), ("I", "l"), ("O", "0"), ("o", "0"), ("e", "c"), ("a", "o"), ("S", "5"), ("rn", "m"),
        ("m", "rn"), ("h", "b"), ("u", "v"), ("t", "f"), ("i", "í"), ("c", "e"), ("n", "ri"), ("é", "e"),
        ("ü", "u"), ("ä", "a"), ("и", "н"), ("п", "л"), ("е", "с")]


def noise(s: str, rng: random.Random, rate: float = 0.06) -> str:
    """light OCR noise on non-key text: a substituted / dropped / doubled letter or a split word in ~rate of words"""
    out = []
    for w in s.split(" "):
        if len(w) > 3 and rng.random() < rate and not any(ch.isdigit() for ch in w):
            r = rng.random()
            if r < 0.55:
                cands = [(a, b) for a, b in SUBS if a in w]
                if cands:
                    a, b = rng.choice(cands)
                    k = w.find(a)
                    w = w[:k] + b + w[k + len(a):]
            elif r < 0.7:
                k = rng.randrange(1, len(w) - 1)
                w = w[:k] + w[k + 1:]
            elif r < 0.85:
                k = rng.randrange(1, len(w) - 1)
                w = w[:k] + " " + w[k:]
            else:
                k = rng.randrange(1, len(w) - 1)
                w = w[:k] + w[k] + w[k:]
        out.append(w)
    return " ".join(out)


class Doc:
    """one document under construction: consistent per-document number and date styles, the printed dates (for the
    DD/MM vs MM/DD check), and the evidence strings that must appear in the text"""

    def __init__(self, rng: random.Random, loc: str):
        self.rng, self.loc, self.lang = rng, loc, LANG[loc]
        self.thou = rng.random() < 0.75
        self.frsep = rng.choice([" ", " ", " "])
        self.dstyle = rng.choice(DATE_STYLES[loc]) if rng.random() < 0.9 else "Y-m-d"
        self.dstyle2 = rng.choice(DATE_STYLES[loc]) if rng.random() < 0.3 else self.dstyle   # mixed formats
        self.printed = []                       # (style, date)
        self.ev = []                            # evidence strings
        self.currency = CUR[loc]
        self.cur_style = rng.random()

    # numbers
    def amt(self, x, sym: bool = False) -> str:
        x = D(x)
        neg = x < 0
        ip, fp = f"{abs(x):.2f}".split(".")
        if self.thou and len(ip) > 3:
            g = f"{int(ip):,}"
            ip = g if self.lang == "en" else g.replace(",", ".") if self.loc == "DE" else g.replace(",", self.frsep)
        s = ip + ("." if self.lang == "en" else ",") + fp
        if neg:
            s = "-" + s
        return self.cur(s) if sym else s

    def cur(self, s: str) -> str:
        c, r = self.currency, self.cur_style
        if c == "USD":
            return "$" + s if not s.startswith("-") else "-$" + s[1:]
        if c == "GBP":
            return "£" + s if not s.startswith("-") else "-£" + s[1:]
        if c == "BGN":
            return s + " лв." if r < 0.7 else "BGN " + s
        if c == "EUR":
            if self.lang == "en":
                return ("€" + s) if r < 0.6 else ("EUR " + s)
            return (s + " €") if r < 0.6 else (s + " EUR")
        return f"{c} {s}"

    # dates
    def date(self, d: dt.date, style: str | None = None, record: bool = True) -> str:
        st, mon, en = style or self.dstyle, MONTHS[self.lang], MONTHS["en"]
        dd, mm, y = d.day, d.month, d.year
        s = {"Y-m-d": d.isoformat(),
             "m/d/Y": f"{mm:02d}/{dd:02d}/{y}", "m/d/y": f"{mm:02d}/{dd:02d}/{y % 100:02d}",
             "m-d-Y": f"{mm:02d}-{dd:02d}-{y}", "Mon d, Y": f"{en[mm - 1][:3]} {dd}, {y}",
             "Month d, Y": f"{en[mm - 1]} {dd}, {y}",
             "d/m/Y": f"{dd:02d}/{mm:02d}/{y}", "d/m/y": f"{dd:02d}/{mm:02d}/{y % 100:02d}",
             "d Month Y": f"{dd} {mon[mm - 1]} {y}", "d Mon Y": f"{dd} {mon[mm - 1][:3]} {y}",
             "dd-Mon-Y": f"{dd:02d}-{mon[mm - 1][:3]}-{y}",
             "d.m.Y": f"{dd:02d}.{mm:02d}.{y}", "d.m.y": f"{dd:02d}.{mm:02d}.{y % 100:02d}",
             "d. Month Y": f"{dd}. {mon[mm - 1]} {y}",
             "d1 month Y": f"{'1er' if dd == 1 else dd} {mon[mm - 1]} {y}",
             "d.m.Y г.": f"{dd:02d}.{mm:02d}.{y} г.", "d month Y г.": f"{dd} {mon[mm - 1]} {y} г."}[st]
        if record:
            self.printed.append((st, d))
        return s

    def date_note(self) -> str | None:
        """a format note when some EN numeric date is ambiguous and no date in the same style disambiguates it"""
        for st, d in self.printed:
            if self.lang == "en" and st in AMBIG and d.day <= 12 and d.day != d.month:
                if not any(s2 == st and d2.day > 12 for s2, d2 in self.printed):
                    return {"m": "MM/DD/YYYY", "d": "DD/MM/YYYY"}[st[0]]
        return None


def rand_date(rng, a=dt.date(2024, 1, 2), b=dt.date(2026, 9, 28)) -> dt.date:
    return a + dt.timedelta(days=rng.randrange((b - a).days))


def company(rng, loc, avoid=()):
    while True:
        if LANG[loc] == "en":
            suf = {"US": ["Inc.", "LLC", "Corp.", "Co."], "UK": ["Ltd", "Limited", "LLP", "plc"],
                   "IE": ["Ltd", "Limited", "DAC"]}[loc]
            n = f"{rng.choice(EN_A)} {rng.choice(EN_B)} {rng.choice(suf)}"
        elif loc == "DE":
            n = f"{rng.choice(DE_A)} {rng.choice(DE_B)} {rng.choice(['GmbH', 'AG', 'GmbH & Co. KG', 'KG', 'OHG'])}"
        elif loc == "FR":
            n = f"{rng.choice(FR_B)} {rng.choice(FR_A)} {rng.choice(['SARL', 'SAS', 'SA', 'EURL'])}"
        else:
            n = f"„{rng.choice(BG_B)} {rng.choice(BG_A)}“ {rng.choice(['ЕООД', 'ООД', 'АД', 'ЕАД'])}"
        if n not in avoid and all(n.split()[0] != a.split()[0] for a in avoid):
            return n


def person(rng, loc):
    """(first, middle, last); middle only for Bulgarian names (the father's name)"""
    if loc == "BG":
        male = rng.random() < 0.5
        f = rng.choice(BG_M if male else BG_F)
        return f, rng.choice(BG_PAT) + ("" if male else "а"), rng.choice(BG_LAST) + ("" if male else "а")
    k = loc if loc in ("DE", "FR") else "US"
    return rng.choice(FIRST[k]), "", rng.choice(LAST[k])


def full(p):
    return " ".join(x for x in p if x)


def address(rng, loc):
    """dict(street, postal, city, lines, country)"""
    if loc == "US":
        city, state, z = rng.choice(US_CITY)
        st = f"{rng.randint(10, 9899)} {rng.choice(['Maple', 'Oak', 'Cedar', 'Lakeview', 'Washington', 'Lincoln', 'Park', 'Elm', 'Highland', 'Sunset'])} {rng.choice(['Street', 'Avenue', 'Road', 'Drive', 'Boulevard', 'Lane'])}"
        return dict(street=st, postal=z, city=city, lines=[st, f"{city}, {state} {z}"])
    if loc == "UK":
        city, pc = rng.choice(UK_CITY)
        st = f"{rng.randint(1, 240)} {rng.choice(['Harbour', 'Victoria', 'Church', 'Mill', 'Station', 'Queen', 'High', 'Bridge', 'Castle', 'Albion'])} {rng.choice(['Road', 'Street', 'Lane', 'Way', 'Avenue', 'Crescent'])}"
        return dict(street=st, postal=pc, city=city,
                    lines=[st, city, pc] if rng.random() < 0.5 else [st, f"{city} {pc}"])
    if loc == "IE":
        city, pc = rng.choice(IE_CITY)
        st = f"{rng.randint(1, 120)} {rng.choice(['Dame', 'Grafton', 'Patrick', 'Castle', 'Main', 'Shop'])} Street"
        return dict(street=st, postal=pc, city=city, lines=[st, f"{city} {pc}"])
    if loc == "DE":
        pc, city = rng.choice(DE_CITY)
        st = f"{rng.choice(['Haupt', 'Bahnhof', 'Garten', 'Schiller', 'Goethe', 'Linden', 'Berg', 'Kirch', 'Mühlen', 'Wald'])}{rng.choice(['straße', 'weg', 'allee', 'gasse'])} {rng.randint(1, 140)}{rng.choice(['', '', '', 'a', 'b'])}"
        return dict(street=st, postal=pc, city=city, lines=[st, f"{pc} {city}"])
    if loc == "FR":
        pc, city = rng.choice(FR_CITY)
        st = f"{rng.randint(1, 160)} {rng.choice(['rue', 'avenue', 'boulevard', 'allée', 'impasse'])} {rng.choice(['de la Paix', 'Victor Hugo', 'des Lilas', 'du Général Leclerc', 'Jean Jaurès', 'de la République', 'Pasteur', 'des Écoles', 'Saint-Michel', 'du Moulin'])}"
        return dict(street=st, postal=pc, city=city,
                    lines=[st, f"{pc} {city.upper() if rng.random() < 0.4 else city}"])
    pc, city = rng.choice(BG_CITY)
    st = (f"ул. „{rng.choice(['Витоша', 'Шипка', 'Раковски', 'Граф Игнатиев', 'Цар Симеон', 'Васил Левски', 'Христо Ботев', 'Оборище', 'Солунска'])}“ № {rng.randint(1, 120)}"
          if rng.random() < 0.7 else
          f"бул. „{rng.choice(['Витоша', 'Цариградско шосе', 'България', 'Христо Ботев', 'Скобелев'])}“ {rng.randint(1, 120)}")
    return dict(street=st, postal=pc, city=city,
                lines=[st, f"{pc} {city}"] if rng.random() < 0.5 else [f"гр. {city} {pc}", st])


def two_col(left, right, rng, gap=None):
    """OCR of two side-by-side blocks: lines merged with uneven spacing"""
    w = max([len(x) for x in left] + [10]) + (gap if gap is not None else rng.randint(4, 16))
    out = []
    for i in range(max(len(left), len(right))):
        a = left[i] if i < len(left) else ""
        b = right[i] if i < len(right) else ""
        out.append((a + " " * max(2, w - len(a) + rng.randint(-3, 3)) + b).rstrip() if b else a)
    return out


def table(rows, rng, widths=None, jitter=2, left=1):
    """rows of cells -> lines with roughly aligned columns (the first `left` columns left-aligned, the rest
    right-aligned; widths jittered)"""
    n = max(len(r) for r in rows)
    widths = widths or [max(len(str(r[i])) for r in rows if i < len(r)) + 2 for i in range(n)]
    out = []
    for r in rows:
        s = ""
        for i, cell in enumerate(r):
            cell = str(cell)
            w = widths[i] + rng.randint(-jitter, jitter)
            s += (cell.ljust(w) if i < left else cell.rjust(w)) + " "
        out.append(s.rstrip())
    return out


def num_word(n, lang):
    words = {"en": {1: "one", 2: "two", 3: "three", 4: "four", 6: "six", 12: "twelve", 14: "fourteen", 30: "thirty",
                    60: "sixty", 90: "ninety", 8: "eight"},
             "de": {1: "einem", 2: "zwei", 3: "drei", 4: "vier", 6: "sechs", 12: "zwölf", 14: "vierzehn",
                    30: "dreißig", 60: "sechzig", 90: "neunzig", 8: "acht"},
             "fr": {1: "un", 2: "deux", 3: "trois", 4: "quatre", 6: "six", 12: "douze", 14: "quatorze",
                    30: "trente", 60: "soixante", 90: "quatre-vingt-dix", 8: "huit"}}
    return words.get(lang, {}).get(n, str(n))


# ---- invoices -------------------------------------------------------------------------------------------------------
INV = {
    "en": dict(title=["INVOICE", "Invoice", "TAX INVOICE", "SALES INVOICE"],
               no=["Invoice No.", "Invoice #", "Invoice Number", "Inv. No."],
               date=["Invoice Date", "Date of Issue", "Invoice date", "Date"],
               due=["Due Date", "Payment Due", "Due date", "Please pay by"],
               odate=["Order Date", "Date of Order", "Order date"], ddate=["Delivery Date", "Ship Date", "Date of Supply"],
               po=["PO Number", "Purchase Order", "Your PO No.", "Customer PO"],
               cno=["Customer No.", "Account No.", "Customer ID"], dn=["Delivery Note No.", "Packing Slip No."],
               bill=["Bill To", "Invoice To", "Billed To", "Customer"], ship=["Ship To", "Deliver To"],
               cols=[("Description", "Qty", "Unit Price", "Amount"), ("Item", "Quantity", "Rate", "Line Total"),
                     ("Details", "Units", "Price", "Total")],
               sub=["Subtotal", "Net Total", "Sub-total"], vat="VAT {r}%", ustax="Sales Tax ({r}%)",
               total=["Total", "Invoice Total", "TOTAL", "Grand Total"], total2=["Invoice Total", "Total this invoice"],
               terms=["Payment terms: {n} days net", "Terms: Net {n}", "Payment due within {n} days of invoice date"],
               prev="Previous balance", paid="Payment received {d} - thank you", fwd="Balance brought forward",
               amount_due=["Amount Due", "Total amount due", "Balance Due"], dep="Less deposit received",
               rc="VAT reverse charge: the customer accounts for VAT (Art. 196 Directive 2006/112/EC)",
               exempt="Sales Tax (exempt, resale certificate on file)", cur_note="All amounts in {c}",
               thanks=["Thank you for your business!", "Thank you for your order.", "We appreciate your prompt payment."],
               tc=["Goods remain the property of the seller until paid in full.",
                   "Interest may be charged on overdue accounts at 8% per annum above base rate.",
                   "Please quote the invoice number with your payment.",
                   "Any discrepancies must be reported within 7 days of receipt of goods.",
                   "Returns are accepted within 30 days in original packaging only."],
               items=[("Consulting services (hours)", 85, 220, "h"), ("A4 copier paper, box of 5 reams", 18, 32, "u"),
                      ("Ink cartridge 305XL black", 19, 45, "u"), ("Network switch 24-port", 140, 480, "u"),
                      ("Installation labour (hours)", 45, 95, "h"), ("Annual software licence", 300, 4800, "1"),
                      ("Ergonomic office chair", 180, 650, "u"), ("Printed brochures A5, 4pp (per 100)", 35, 90, "u"),
                      ("Office cleaning services (monthly)", 350, 1800, "1"), ("Pallet wrap 500mm x 300m", 9, 22, "u"),
                      ("Laptop stand, aluminium", 25, 70, "u"), ("Website hosting (12 months)", 120, 900, "1"),
                      ("Safety gloves, size 9 (pair)", 2.5, 9, "u"), ("Copper pipe 15mm x 3m", 12, 28, "u"),
                      ("Catering: lunch for 20", 220, 480, "1"), ("Translation (per 1,000 words)", 60, 140, "u"),
                      ("Cable ties 300mm, pack of 100", 4, 12, "u"), ("LED panel 600x600", 25, 75, "u"),
                      ("Training workshop (1 day)", 650, 2400, "1"), ("Warehouse storage, pallet/week", 4, 12, "u")],
               ship_item="Delivery charge", page="Page 1 of 1"),
    "de": dict(title=["RECHNUNG", "Rechnung"], no=["Rechnungsnummer", "Rechnungs-Nr.", "Rechnung Nr."],
               date=["Rechnungsdatum", "Datum"], due=["Fällig am", "Zahlbar bis", "Fälligkeitsdatum"],
               odate=["Bestelldatum", "Auftragsdatum"], ddate=["Lieferdatum", "Leistungsdatum"],
               po=["Ihre Bestellnummer", "Ihre Bestell-Nr.", "Bestellnummer Kunde"], cno=["Kundennummer", "Kunden-Nr."],
               dn=["Lieferschein-Nr."], bill=["Rechnungsempfänger", "An"], ship=["Lieferanschrift"],
               cols=[("Bezeichnung", "Menge", "Einzelpreis", "Gesamt"), ("Artikel", "Anz.", "E-Preis", "Betrag")],
               sub=["Nettobetrag", "Zwischensumme netto", "Summe netto"], vat="MwSt. {r} %", ustax="MwSt. {r} %",
               total=["Gesamtbetrag", "Rechnungsbetrag", "Endbetrag", "Summe brutto"],
               total2=["Rechnungsbetrag", "Gesamtbetrag dieser Rechnung"],
               terms=["Zahlbar innerhalb von {n} Tagen netto", "Zahlungsziel: {n} Tage"],
               prev="Saldo aus Vorrechnung", paid="Zahlungseingang {d}, vielen Dank", fwd="Saldovortrag",
               amount_due=["Offener Betrag gesamt", "Zu zahlender Betrag"], dep="abzgl. Anzahlung",
               rc="Steuerschuldnerschaft des Leistungsempfängers (Reverse-Charge-Verfahren)",
               exempt="", cur_note="Alle Beträge in {c}",
               thanks=["Vielen Dank für Ihren Auftrag!", "Wir danken für Ihre Bestellung."],
               tc=["Die Ware bleibt bis zur vollständigen Bezahlung unser Eigentum.",
                   "Es gelten unsere Allgemeinen Geschäftsbedingungen.",
                   "Bitte geben Sie bei Zahlung die Rechnungsnummer an.",
                   "Reklamationen bitte innerhalb von 8 Tagen nach Erhalt der Ware."],
               items=[("Beratungsleistung (Stunden)", 80, 190, "h"), ("Kopierpapier A4, 5 Pakete", 16, 29, "u"),
                      ("Druckerpatrone schwarz", 18, 42, "u"), ("Montage (Stunden)", 45, 85, "h"),
                      ("Softwarelizenz (12 Monate)", 290, 3900, "1"), ("Bürostuhl ergonomisch", 170, 620, "u"),
                      ("Flyer A5 beidseitig (100 St.)", 25, 70, "u"), ("Reinigung Büroflächen (Monat)", 320, 1600, "1"),
                      ("Kabelbinder 300 mm (100 St.)", 4, 11, "u"), ("LED-Panel 62x62", 22, 70, "u"),
                      ("Schulung (1 Tag)", 600, 2200, "1"), ("Kupferrohr 15 mm, 3 m", 11, 26, "u")],
               items7=[("Fachbuch Steuerrecht 2025", 39, 89, "u"), ("Kaffee, Bohne 1 kg", 14, 26, "u"),
                       ("Fachzeitschrift (Jahresabo)", 120, 340, "1")],
               ship_item="Versandkosten", page="Seite 1 von 1"),
    "fr": dict(title=["FACTURE", "Facture"], no=["Facture n°", "N° de facture", "Numéro de facture"],
               date=["Date de facture", "Date d'émission", "Date"], due=["Date d'échéance", "Échéance", "À régler avant le"],
               odate=["Date de commande"], ddate=["Date de livraison"],
               po=["Votre bon de commande n°", "Réf. commande client", "N° de commande client"],
               cno=["N° client", "Code client"], dn=["Bon de livraison n°"], bill=["Facturé à", "Client"],
               ship=["Livré à"], cols=[("Désignation", "Qté", "P.U. HT", "Montant HT"),
                                       ("Description", "Quantité", "Prix unitaire HT", "Total HT")],
               sub=["Total HT", "Montant HT"], vat="TVA {r} %", ustax="TVA {r} %",
               total=["Total TTC", "Montant TTC"], total2=["Total TTC de la facture"],
               terms=["Paiement à {n} jours", "Conditions de paiement : {n} jours net"],
               prev="Solde antérieur", paid="Règlement reçu le {d}, merci", fwd="Report de solde",
               amount_due=["Net à payer", "Reste à payer"], dep="Acompte versé",
               rc="Autoliquidation : TVA due par le preneur (art. 283-2 du CGI)", exempt="",
               cur_note="Montants exprimés en {c}", thanks=["Merci de votre confiance."],
               tc=["En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due.",
                   "Pas d'escompte pour paiement anticipé.",
                   "Réserve de propriété : les marchandises restent notre propriété jusqu'au paiement intégral."],
               items=[("Prestation de conseil (heures)", 80, 190, "h"), ("Papier A4, carton de 5 ramettes", 17, 30, "u"),
                      ("Cartouche d'encre noire", 18, 44, "u"), ("Main-d'œuvre installation (heures)", 42, 85, "h"),
                      ("Licence logicielle annuelle", 290, 4200, "1"), ("Fauteuil de bureau", 160, 590, "u"),
                      ("Dépliants A5 recto-verso (le cent)", 28, 75, "u"), ("Nettoyage des locaux (mois)", 300, 1500, "1"),
                      ("Formation (1 journée)", 600, 2100, "1"), ("Câble réseau 10 m", 6, 19, "u"),
                      ("Écran 27 pouces", 190, 420, "u")],
               items7=[("Ouvrage technique (livre)", 29, 79, "u"), ("Abonnement revue professionnelle", 90, 290, "1")],
               ship_item="Frais de port", page="Page 1/1"),
    "bg": dict(title=["ФАКТУРА", "Фактура"], no=["Фактура №", "Номер на фактура"], date=["Дата на издаване", "Дата"],
               due=["Срок за плащане", "Падеж", "Дата на падеж"], odate=["Дата на поръчка"],
               ddate=["Дата на данъчното събитие", "Дата на доставка"], po=["Поръчка на клиента №", "Ваша поръчка №"],
               cno=["Клиентски №"], dn=["Стокова разписка №"], bill=["Получател", "Клиент"], ship=["Адрес на доставка"],
               cols=[("Наименование", "Кол.", "Ед. цена", "Стойност"), ("Описание", "Количество", "Единична цена", "Сума")],
               sub=["Данъчна основа", "Сума без ДДС", "Общо без ДДС"], vat="ДДС {r}%", ustax="ДДС {r}%",
               total=["Сума за плащане", "Обща сума", "Всичко"], total2=["Обща сума по фактурата"],
               terms=["Плащане до {n} дни от датата на издаване"],
               prev="Предходно салдо", paid="Получено плащане на {d}", fwd="Пренесено салдо",
               amount_due=["Общо дължимо", "Остатък за плащане"], dep="Приспаднат аванс",
               rc="Основание за неначисляване на ДДС: чл. 163а от ЗДДС (обратно начисляване)", exempt="",
               cur_note="Всички суми са в {c}", thanks=["Благодарим Ви!"],
               tc=["Съгласно чл. 6, ал. 1 от Закона за счетоводството, подписът и печатът не са задължителни реквизити.",
                   "Моля, посочете номера на фактурата при плащане."],
               items=[("Консултантски услуги (часове)", 120, 380, "h"), ("Копирна хартия А4, 5 пакета", 32, 58, "u"),
                      ("Тонер касета черна", 45, 160, "u"), ("Монтаж (часове)", 50, 120, "h"),
                      ("Софтуерен лиценз (12 месеца)", 600, 7800, "1"), ("Офис стол", 260, 990, "u"),
                      ("Флаери А5 (100 бр.)", 40, 120, "u"), ("Почистване на офис (месец)", 450, 2600, "1"),
                      ("Обучение (1 ден)", 900, 3800, "1"), ("Мрежов кабел 10 м", 12, 35, "u")],
               ship_item="Транспортни разходи", page="Стр. 1 от 1"),
}
TAXID = {"US": lambda r: f"EIN {r.randint(10, 99)}-{r.randint(1000000, 9999999)}",
         "UK": lambda r: f"VAT Reg. No. GB {r.randint(100, 999)} {r.randint(1000, 9999)} {r.randint(10, 99)}",
         "IE": lambda r: f"VAT No. IE{r.randint(1000000, 9999999)}{r.choice('ABCDEFGHW')}",
         "DE": lambda r: f"USt-IdNr.: DE{r.randint(100000000, 999999999)}",
         "FR": lambda r: f"SIRET {r.randint(100, 999)} {r.randint(100, 999)} {r.randint(100, 999)} 000{r.randint(10, 99)} - TVA FR{r.randint(10, 99)}{r.randint(100000000, 999999999)}",
         "BG": lambda r: f"ЕИК: {r.randint(100000000, 999999999)}, ДДС №: BG{r.randint(100000000, 999999999)}"}


def phone(rng, loc):
    r = rng.randint
    return {"US": f"({r(201, 989)}) {r(200, 999)}-{r(1000, 9999)}", "UK": f"+44 {r(20, 29)} {r(1000, 9999)} {r(1000, 9999)}",
            "IE": f"+353 {r(1, 99)} {r(100, 999)} {r(1000, 9999)}", "DE": f"+49 {r(30, 899)} {r(100000, 9999999)}",
            "FR": f"+33 {r(1, 5)} {r(10, 99)} {r(10, 99)} {r(10, 99)} {r(10, 99)}",
            "BG": f"+359 {r(2, 99)} {r(100, 999)} {r(100, 9999)}"}[loc]


def bank_line(rng, loc):
    if loc == "US":
        return f"Remit by ACH: routing no. 0{rng.randint(10000000, 99999999)}, account no. {rng.randint(100000000, 999999999)}"
    if loc == "UK":
        return f"Bank: Northgate Savings Bank  Sort code {rng.randint(10, 99)}-{rng.randint(10, 99)}-{rng.randint(10, 99)}  Account {rng.randint(10000000, 99999999)}"
    cc = {"IE": "IE", "DE": "DE", "FR": "FR", "BG": "BG"}[loc]
    digits = "".join(str(rng.randint(0, 9)) for _ in range(18))
    iban = f"{cc}{rng.randint(10, 99)} " + " ".join(digits[i:i + 4] for i in range(0, 18, 4))
    lab = {"IE": "IBAN", "DE": "Bankverbindung: IBAN", "FR": "IBAN", "BG": "IBAN:"}[loc]
    return f"{lab} {iban}  BIC {''.join(rng.choice('ABCDEFGHKLMNPRSTUVWXZ') for _ in range(4))}{cc}{rng.choice(['2X', 'FF', 'PP', 'SD'])}"


def doc_id_number(rng, loc, year):
    n = rng.randint(1, 9999)
    if loc == "DE":
        return rng.choice([f"RE-{year}-{n:05d}", f"{year}{n:04d}", f"R{rng.randint(100000, 999999)}"])
    if loc == "FR":
        return rng.choice([f"FA{year}-{n:04d}", f"F-{year}-{n:05d}"])
    if loc == "BG":
        return f"{rng.randint(1000, 99999):010d}"
    return rng.choice([f"INV-{year}-{n:04d}", f"{rng.randint(100000, 999999)}", f"INV{rng.randint(10000, 99999)}",
                       f"{rng.choice(['AC', 'NW', 'KS', 'BR'])}{year % 100:02d}{n:04d}"])


def gen_invoice(d: Doc):
    rng, loc, lang = d.rng, d.loc, d.lang
    L = INV[lang]
    vendor = company(rng, loc)
    cloc = loc
    if loc in ("IE", "DE", "FR") and rng.random() < 0.2:
        cloc = rng.choice([x for x in ("IE", "DE", "FR") if x != loc])
    customer = company(rng, cloc, avoid=(vendor,))
    va, ca = address(rng, loc), address(rng, cloc)
    if loc == "UK" and rng.random() < 0.2:
        d.currency = rng.choice(["EUR", "USD"])
    inv = rand_date(rng)
    odate = inv - dt.timedelta(days=rng.randint(2, 30)) if rng.random() < 0.65 else None
    ddate = inv - dt.timedelta(days=rng.randint(0, 7)) if rng.random() < 0.4 else None
    due_mode = rng.choices(["date", "terms", "none"], [0.6, 0.3, 0.1])[0]
    nterm = rng.choice([14, 30, 30, 45, 60])
    due = inv + dt.timedelta(days=nterm) if due_mode == "date" else None
    po = (rng.choice([f"PO-{rng.randint(10000, 99999)}", f"45{rng.randint(10000000, 99999999)}",
                      f"PO{inv.year}/{rng.randint(100, 9999):04d}"]) if rng.random() < 0.6 else None)
    cno = f"{rng.choice(['C', 'K', 'CU', ''])}{rng.randint(10000, 99999)}" if rng.random() < 0.7 else None
    dn = f"{rng.choice(['DN', 'LS', 'BL', ''])}{rng.randint(100000, 999999)}" if rng.random() < 0.3 else None
    invno = doc_id_number(rng, loc, inv.year)

    # items, tax
    lines, bases = [], {}
    rate_std = {"US": rng.choice([Decimal("6.25"), Decimal("7"), Decimal("7.25"), Decimal("8"), Decimal("8.25"),
                                  Decimal("8.875")]),
                "UK": Decimal(20), "IE": Decimal(23), "DE": Decimal(19), "FR": Decimal(20), "BG": Decimal(20)}[loc]
    low = {"DE": Decimal(7), "FR": Decimal("5.5")}.get(loc)
    for k in range(rng.randint(2, 6)):
        if low and rng.random() < 0.25:
            (desc, lo, hi, qk), rate = rng.choice(L["items7"]), low
        else:
            (desc, lo, hi, qk), rate = rng.choice(L["items"]), rate_std
        qty = (Decimal(rng.randint(2, 40)) / 2 if qk == "h" else Decimal(1) if qk == "1" else
               Decimal(rng.choice([1, 1, 2, 3, 4, 5, 6, 8, 10, 12, 20, 25, 50])))
        price = D(rng.uniform(lo, hi))
        if rng.random() < 0.4:
            price = D(round(float(price)) if hi > 30 else round(float(price) * 2) / 2)
        amount = D(qty * price)
        lines.append((desc, qty, price, amount))
        bases[rate] = bases.get(rate, Decimal(0)) + amount
    if rng.random() < 0.3:
        sc = D(rng.choice([4.95, 9.5, 12, 15, 25, 39.9]))
        lines.append((L["ship_item"], Decimal(1), sc, sc))
        bases[rate_std] = bases.get(rate_std, Decimal(0)) + sc
    subtotal = sum(a for *_, a in lines)
    reverse = cloc != loc and rng.random() < 0.8
    exempt = loc == "US" and rng.random() < 0.1
    taxlines = []
    if reverse or exempt:
        taxlines.append((L["rc"] if reverse else L["exempt"], Decimal(0)))
    else:
        for r in sorted(bases):
            lab = (L["ustax"] if loc == "US" else L["vat"]).format(r=f"{r:g}".replace(".", "," if lang != "en" else "."))
            if len(bases) > 1:
                lab += {"en": " on {b}", "de": " auf {b}", "fr": " sur {b}", "bg": " върху {b}"}[lang].format(b=d.amt(bases[r]))
            taxlines.append((lab, D(bases[r] * r / 100)))
    tax = sum(t for _, t in taxlines)
    total = subtotal + tax

    # extra lines after the total (distractors)
    extra = rng.random() < (0.25 if lang == "en" else 0.12)
    prev = paid = dep = None
    if extra and rng.random() < 0.6:
        prev = D(rng.uniform(80, 2500))
        paid = D(prev * Decimal(rng.choice([1, 1, 0.5, 0.75])))
    elif extra:
        dep = D(total * Decimal(rng.choice([0.2, 0.3, 0.5])))

    # header
    tl = rng.choice(L["title"])
    meta = [(rng.choice(L["no"]), invno), (rng.choice(L["date"]), d.date(inv))]
    if due:
        meta.append((rng.choice(L["due"]), d.date(due, d.dstyle2)))
    if odate:
        meta.insert(rng.randint(1, len(meta)), (rng.choice(L["odate"]), d.date(odate, d.dstyle2)))
    if ddate:
        meta.append((rng.choice(L["ddate"]), d.date(ddate)))
    if po:
        meta.append((rng.choice(L["po"]), po))
    if cno:
        meta.insert(rng.randint(0, len(meta)), (rng.choice(L["cno"]), cno))
    if dn:
        meta.append((rng.choice(L["dn"]), dn))
    vblock = [vendor if rng.random() < 0.6 else vendor.upper()] + va["lines"] + (
        [rng.choice(COUNTRY_PRINT[loc])] if rng.random() < 0.3 else [])
    vblock.append(rng.choice([f"Tel. {phone(rng, loc)}", TAXID[loc](rng)]))
    bill = [rng.choice(L["bill"]) + ":", customer] + ca["lines"] + (
        [rng.choice(COUNTRY_PRINT[cloc])] if cloc != loc else [])
    layout = rng.choice(["two", "stack", "split"])
    out = []
    if layout == "two":
        right = [tl, ""] + [f"{k}: {v}" for k, v in meta]
        out += two_col(vblock, right, rng)
    else:
        out += vblock + ["", tl.center(40).rstrip() if rng.random() < 0.5 else tl, ""]
        if layout == "stack":
            pairs = [f"{k}: {v}" for k, v in meta]
            while pairs:
                if len(pairs) > 1 and rng.random() < 0.4:
                    out.append(pairs[0] + " " * rng.randint(3, 12) + pairs[1])
                    pairs = pairs[2:]
                else:
                    out.append(pairs.pop(0))
        else:                                            # labels, then values (column-major OCR of a label table)
            out += [k for k, _ in meta] + [""] + [v for _, v in meta]
    out.append("")
    if rng.random() < 0.3:
        ship = [rng.choice(L["ship"]) + ":", rng.choice([customer, company(rng, cloc, avoid=(vendor, customer))])] + address(rng, cloc)["lines"]
        out += two_col(bill, ship, rng)
    else:
        out += bill
    out.append("")
    if d.currency != CUR[loc]:
        out.append(L["cur_note"].format(c=d.currency))
    # items table
    cols = rng.choice(L["cols"])
    rows = [list(cols)]
    for desc, qty, price, amount in lines:
        q = f"{qty:g}" if lang == "en" else f"{qty:g}".replace(".", ",")
        if len(desc) > 26 and rng.random() < 0.35:              # wrapped description
            cut = desc.rfind(" ", 0, 26)
            rows.append([desc[:cut], q, d.amt(price), d.amt(amount)])
            rows.append(["  " + desc[cut + 1:]])
        else:
            rows.append([desc, q, d.amt(price), d.amt(amount)])
    out += table(rows, rng, jitter=3 if rng.random() < 0.5 else 1)
    out.append("-" * rng.randint(30, 60))
    tot_rows = [[rng.choice(L["sub"]), d.amt(subtotal, sym=rng.random() < 0.3)]]
    for lab, t in taxlines:
        tot_rows.append([lab, d.amt(t)])
    tlab = rng.choice(L["total2"] if (prev or dep) else L["total"])
    tot_rows.append([tlab, d.amt(total, sym=True)])
    if prev is not None:
        fwd = prev - paid
        pd = inv - dt.timedelta(days=rng.randint(5, 25))
        tot_rows += [[L["prev"], d.amt(prev)], [L["paid"].format(d=d.date(pd)), d.amt(-paid)],
                     [L["fwd"], d.amt(fwd)], [rng.choice(L["amount_due"]), d.amt(total + fwd, sym=True)]]
    if dep is not None:
        tot_rows += [[L["dep"], d.amt(-dep)], [rng.choice(L["amount_due"]), d.amt(total - dep, sym=True)]]
    ind = " " * rng.randint(4, 30)
    out += [ind + x for x in table(tot_rows, rng, jitter=1)]
    out.append("")
    if due_mode == "terms" or (due_mode == "date" and rng.random() < 0.4):
        out.append(rng.choice(L["terms"]).format(n=nterm))
    out.append(noise(rng.choice(L["thanks"]), rng))
    out += [noise(x, rng) for x in rng.sample(L["tc"], k=min(len(L["tc"]), rng.randint(1, 3)))]
    out.append(bank_line(rng, loc))
    out.append(noise(TAXID[loc](rng) if rng.random() < 0.5 else vendor + " - " + ", ".join(va["lines"]), rng, 0.02))
    out.append(L["page"])
    gold = dict(vendor=vendor, invoice_number=invno, invoice_date=inv.isoformat(),
                due_date=due.isoformat() if due else None, currency=d.currency, subtotal=float(subtotal),
                tax=float(tax), total=float(total), po_number=po)
    d.ev_ci = [vendor]
    d.ev += [invno, d.amt(subtotal), d.amt(total)] + ([po] if po else []) + (
        [d.amt(tax)] if len(taxlines) == 1 else [])
    return out, gold


# ---- receipts -------------------------------------------------------------------------------------------------------
RCP = {
    "en": dict(shops=[("FRESHMART", "shop"), ("Corner Deli & Grocer", "shop"), ("Bean There Coffee", "food"),
                      ("Hilltop Hardware", "shop"), ("The Golden Fork", "food"), ("PageTurner Books", "shop"),
                      ("Greenleaf Pharmacy", "shop"), ("Urban Bakehouse", "food"), ("Sunrise Diner", "food"),
                      ("Harbourside Fish Bar", "food"), ("Brightside Garden Centre", "shop")],
               shop=[("Milk 2L", 1.89), ("Bread wholemeal", 2.35), ("Bananas 1kg", 1.29), ("Coffee beans 500g", 8.99),
                     ("AA batteries 8pk", 6.49), ("Screws 4x40 (100)", 4.75), ("Paracetamol 500mg x16", 2.10),
                     ("Notebook A5", 3.20), ("Paint brush 50mm", 5.95), ("Potting compost 40L", 7.50),
                     ("Shampoo 400ml", 4.25), ("Eggs (12)", 3.60), ("Orange juice 1L", 2.15), ("Paperback novel", 9.99)],
               food=[("Flat white", 3.40), ("Cappuccino", 3.20), ("Croissant", 2.60), ("Club sandwich", 9.50),
                     ("Soup of the day", 6.25), ("Fish & chips", 12.95), ("Burger + fries", 13.50),
                     ("Caesar salad", 10.75), ("Sparkling water", 2.50), ("House red (glass)", 6.90),
                     ("Cheesecake", 5.80), ("Pancake stack", 8.40)],
               sub=["SUBTOTAL", "Subtotal", "Sub Total"], total=["TOTAL", "Total", "AMOUNT DUE", "TOTAL TO PAY"],
               tip=["Tip", "Gratuity"], service="Service charge 12.5%", cash=["CASH", "Cash tendered"],
               change=["CHANGE", "Change due"], card=["VISA", "MASTERCARD", "DEBIT CARD", "CARD"],
               mobile=["APPLE PAY", "GOOGLE PAY"], loyalty=["Loyalty card", "Rewards member"],
               auth="AUTH CODE", tid="Terminal ID", rno=["Receipt No.", "Trans #", "Txn"], store="Store",
               cashier=["Cashier", "Server", "Operator"], items="Items", saved="You saved",
               vat_incl="VAT {r}% included", ustax="Sales Tax {r}%",
               thanks=["Thank you for shopping with us!", "Please come again", "Keep this receipt for returns within 28 days",
                       "Tell us how we did: survey code on the back"],
               approved=["CARD PAYMENT APPROVED", "APPROVED - THANK YOU"], date="Date", time="Time"),
    "de": dict(shops=[("Bäckerei Kranz", "food"), ("Café Lindenhof", "food"), ("Getränkemarkt Sommer", "shop"),
                      ("Apotheke am Markt", "shop"), ("Buchhandlung Seitenweise", "shop")],
               shop=[("Mineralwasser 6x1,5l", 3.99), ("Apfelsaft 1l", 1.79), ("Ibuprofen 400 20 St.", 4.95),
                     ("Taschenbuch", 12.00), ("Pfand", 1.50), ("Kaffee 500g", 6.49), ("Notizblock A5", 2.49)],
               food=[("Brötchen", 0.45), ("Laugenbrezel", 0.95), ("Kaffee groß", 3.20), ("Cappuccino", 3.40),
                     ("Apfelstrudel", 3.90), ("Belegtes Brötchen", 3.50), ("Kuchen Stück", 2.80)],
               sub=["Zwischensumme"], total=["SUMME", "Gesamt", "Zu zahlen"], tip=["Trinkgeld"],
               service="Bedienung 10%", cash=["Bar", "BAR"], change=["Rückgeld", "Zurück"],
               card=["girocard", "Kartenzahlung", "VISA"], mobile=["Apple Pay", "Google Pay"],
               loyalty=["Kundenkarte"], auth="Autorisierungsnr.", tid="Terminal-ID", rno=["Bon-Nr.", "Beleg-Nr."],
               store="Filiale", cashier=["Kasse", "Bediener"], items="Artikel", saved="Sie sparen",
               vat_incl="enth. MwSt. {r}%", ustax="MwSt. {r}%",
               thanks=["Vielen Dank für Ihren Einkauf!", "Wir freuen uns auf Ihren nächsten Besuch."],
               approved=["Zahlung erfolgt"], date="Datum", time="Uhrzeit"),
    "fr": dict(shops=[("Boulangerie du Marché", "food"), ("Café des Arts", "food"), ("Pharmacie Centrale", "shop"),
                      ("Librairie Le Passage", "shop"), ("Épicerie Fine Rivoli", "shop")],
               shop=[("Eau minérale 1,5 L", 0.85), ("Doliprane 1000 mg", 2.18), ("Roman poche", 8.90),
                     ("Confiture abricot", 4.20), ("Café moulu 250 g", 3.95), ("Carnet A5", 3.50)],
               food=[("Baguette tradition", 1.30), ("Croissant", 1.20), ("Pain au chocolat", 1.35), ("Café", 1.80),
                     ("Café crème", 3.20), ("Tarte aux pommes", 3.60), ("Sandwich jambon-beurre", 4.90)],
               sub=["Sous-total"], total=["TOTAL TTC", "À PAYER", "Total"], tip=["Pourboire"],
               service="Service 10%", cash=["Espèces"], change=["Rendu", "Monnaie rendue"],
               card=["CB", "Carte bancaire", "VISA"], mobile=["Apple Pay", "Google Pay"],
               loyalty=["Carte fidélité"], auth="N° autorisation", tid="N° terminal", rno=["Ticket n°", "N° ticket"],
               store="Magasin", cashier=["Caisse", "Vendeur"], items="Articles", saved="Économie",
               vat_incl="dont TVA {r} %", ustax="TVA {r} %",
               thanks=["Merci de votre visite", "À bientôt !"], approved=["PAIEMENT ACCEPTÉ"], date="Date",
               time="Heure"),
    "bg": dict(shops=[("Пекарна Златно зърно", "food"), ("Кафе Ъгъла", "food"), ("Аптека Здраве", "shop"),
                      ("Книжарница Буквар", "shop"), ("Магазин Свежест", "shop")],
               shop=[("Минерална вода 1,5 л", 0.99), ("Хляб Добруджа", 1.69), ("Аналгин", 2.35), ("Тетрадка А5", 1.80),
                     ("Кафе мляно 250 г", 5.49), ("Кисело мляко", 1.29)],
               food=[("Баница", 1.80), ("Кафе еспресо", 2.20), ("Капучино", 3.50), ("Боза", 1.20),
                     ("Сандвич", 4.50), ("Торта парче", 3.90)],
               sub=["МЕЖДИННА СУМА"], total=["ОБЩА СУМА", "ОБЩО"], tip=["Бакшиш"], service="Обслужване 10%",
               cash=["В БРОЙ"], change=["РЕСТО"], card=["КАРТА", "ДЕБИТНА КАРТА", "VISA"],
               mobile=["Apple Pay", "Google Pay"], loyalty=["Карта за лоялност"], auth="Код за оторизация",
               tid="Терминал", rno=["Бон №", "№"], store="Обект", cashier=["Оператор", "Касиер"], items="Артикули",
               saved="Спестихте", vat_incl="ДДС {r}%", ustax="ДДС {r}%",
               thanks=["Благодарим Ви!", "Заповядайте отново!"], approved=["ОДОБРЕНО"], date="Дата", time="Час"),
}


def fmt_time(rng, loc, h, m, s):
    if loc == "US":
        h12 = h % 12 or 12
        return rng.choice([f"{h12}:{m:02d} {'PM' if h >= 12 else 'AM'}", f"{h12:02d}:{m:02d}{'P' if h >= 12 else 'A'}",
                           f"{h12}:{m:02d}:{s:02d} {'pm' if h >= 12 else 'am'}"])
    return rng.choice([f"{h:02d}:{m:02d}", f"{h:02d}:{m:02d}:{s:02d}"])


def gen_receipt(d: Doc):
    rng, loc, lang = d.rng, d.loc, d.lang
    L = RCP[lang]
    merchant, kind = rng.choice(L["shops"])
    a = address(rng, loc)
    day = rand_date(rng)
    h, mi, se = rng.randint(7, 21), rng.randint(0, 59), rng.randint(0, 59)
    has_time = rng.random() < 0.85
    pool = L["food"] if kind == "food" else L["shop"]
    items = []
    for name, p in rng.sample(pool, min(len(pool), rng.randint(2, 8))):
        q = rng.choice([1, 1, 1, 2, 3])
        items.append((name, q, D(p), D(q * D(p))))
    sub = sum(x[3] for x in items)
    tax = Decimal(0)
    rate = {"US": rng.choice([Decimal("6.25"), Decimal("8.25"), Decimal("7")]), "UK": Decimal(20), "IE": Decimal(23),
            "DE": Decimal(7) if kind == "food" else Decimal(19), "FR": Decimal("5.5") if kind == "food" else Decimal(20),
            "BG": Decimal(20)}[loc]
    if loc == "US":
        tax = D(sub * rate / 100)
    tip = Decimal(0)
    tip_line = None
    if kind == "food" and rng.random() < 0.4:
        if loc == "UK" and rng.random() < 0.5:
            tip = D(sub * Decimal("0.125"))
            tip_line = L["service"]
        else:
            tip = D(rng.choice([1, 2, 3, 5]) if lang != "en" else (sub + tax) * Decimal(rng.choice([0.15, 0.18, 0.2])))
            tip_line = rng.choice(L["tip"])
    total = sub + tax + tip
    pay = rng.choices(["card", "cash", "mobile"], [0.6, 0.3, 0.1])[0]
    last4 = f"{rng.randint(0, 9999):04d}" if pay == "card" else None
    show_last4 = pay == "card" and rng.random() < 0.85
    out = []
    head = [merchant.upper() if rng.random() < 0.5 else merchant]
    if rng.random() < 0.4:
        head.append(f"{L['store']} {rng.randint(10, 999):04d}")
    head += a["lines"]
    head.append(rng.choice([f"Tel {phone(rng, loc)}", TAXID[loc](rng)]))
    width = rng.randint(32, 42)
    out += [x.center(width).rstrip() if rng.random() < 0.6 else x for x in head]
    if lang == "bg":
        out.append("ФИСКАЛЕН БОН".center(width).rstrip())
    out.append("")
    ds = d.date(day)
    ts = fmt_time(rng, loc, h, mi, se) if has_time else ""
    rno = f"{rng.choice(L['rno'])} {rng.randint(100, 99999)}"
    if has_time and rng.random() < 0.5:
        out.append(f"{ds} {ts}" + " " * rng.randint(2, 8) + rno)
    else:
        out.append(f"{L['date']}: {ds}" + (f"   {L['time']}: {ts}" if has_time else ""))
        out.append(rno)
    out.append(f"{rng.choice(L['cashier'])}: {rng.randint(1, 40)}")
    out.append("-" * width)
    for name, q, price, amt in items:
        if q > 1:
            out.append(name)
            out.append(f"  {q} x {d.amt(price)}".ljust(width - 8) + d.amt(amt).rjust(8))
        else:
            out.append(name.ljust(width - 8 + rng.randint(-2, 2)) + d.amt(amt).rjust(8))
    out.append("-" * width)
    rows = []
    if loc == "US" or tip_line or rng.random() < 0.5:
        rows.append([rng.choice(L["sub"]), d.amt(sub)])
    if loc == "US":
        rows.append([L["ustax"].format(r=f"{rate:g}"), d.amt(tax)])
    if tip_line:
        rows.append([tip_line, d.amt(tip)])
    tl = rng.choice(L["total"])
    rows.append([tl, d.amt(total, sym=rng.random() < 0.5)])
    if pay == "cash":
        given = D(max(total, Decimal(5)) // 5 * 5 + rng.choice([5, 10, 20]))
        rows += [[rng.choice(L["cash"]), d.amt(given)], [rng.choice(L["change"]), d.amt(given - total)]]
    elif pay == "card":
        c = rng.choice(L["card"])
        masked = rng.choice([f"************{last4}", f"XXXX XXXX XXXX {last4}", f"**** {last4}", f"xxxx-{last4}"])
        rows.append([c, d.amt(total)])
        if show_last4:
            rows.append([masked, ""])
    else:
        rows.append([rng.choice(L["mobile"]), d.amt(total)])
    out += [r[0].ljust(width - 10) + r[1].rjust(10) if r[1] else r[0] for r in rows]
    if loc != "US":
        vat = D(total - tip - (total - tip) / (1 + rate / 100))
        out.append(L["vat_incl"].format(r=f"{rate:g}".replace(".", "," if lang != "en" else ".")) + "  " + d.amt(vat))
    out.append(f"{L['items']}: {sum(x[1] for x in items)}")
    if pay in ("card", "mobile"):
        out.append(f"{L['auth']}: {rng.randint(0, 999999):06d}   {L['tid']}: {rng.randint(10000000, 99999999)}")
        out.append(rng.choice(L["approved"]))
    if rng.random() < 0.3:
        out.append(f"{rng.choice(L['loyalty'])}: ************{rng.randint(0, 9999):04d}")
    if lang == "de" and rng.random() < 0.6:
        out.append(f"TSE-Transaktion: {rng.randint(10000, 99999)}  TSE-Start: {day.isoformat()}T{(h + rng.randint(0, 1)) % 24:02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}")
    if lang == "bg":
        out.append(f"ИН на ФУ: DT{rng.randint(100000, 999999)}  ФП: {rng.randint(10000000, 99999999)}")
    out.append(noise(rng.choice(L["thanks"]), rng, 0.1))
    gold = dict(merchant=merchant, date=day.isoformat(), time=f"{h:02d}:{mi:02d}" if has_time else None,
                total=float(total), payment_method=pay, card_last4=last4 if show_last4 else None)
    d.ev += [ds, d.amt(total)] + ([last4] if show_last4 else []) + ([ts] if has_time else [])
    return out, gold


# ---- forms ----------------------------------------------------------------------------------------------------------
FORM = {
    "en": dict(kinds=[("Home Insurance Claim Form", "claim", ["Buildings", "Contents", "Accidental damage",
                                                              "Personal possessions", "Legal expenses", "Home emergency"]),
                      ("Motor Insurance Proposal Form", "app", ["Windscreen cover", "Breakdown cover", "Courtesy car",
                                                                "Protected no-claims discount", "Legal cover",
                                                                "Personal accident"]),
                      ("Travel Insurance Application", "app", ["Winter sports", "Cruise cover", "Business travel",
                                                               "Gadget cover", "Cancellation upgrade", "Golf cover"]),
                      ("Health Plan Registration Form", "app", ["Dental", "Optical", "Physiotherapy", "Mental health",
                                                                "Overseas cover", "Hospital cash"])],
               ins=["Insurance plc", "Mutual Insurance", "Assurance Ltd", "General Insurance Co."],
               sec_a="SECTION A - POLICYHOLDER DETAILS", sec_b="SECTION B - ADDRESS",
               sec_c=["SECTION C - COVER REQUIRED (tick all that apply)", "SECTION C - OPTIONAL EXTRAS (tick all that apply)"],
               sec_d="SECTION D - OTHER INFORMATION",
               surname=["Surname", "Last name", "Family name"], first=["First name(s)", "Forename(s)", "Given names"],
               full=["Full name"], lastfirst="Name (Last, First)", title="Title", titles=["Mr", "Mrs", "Ms", "Dr"],
               dob=["Date of birth", "DOB", "Date of Birth"], policy=["Policy number", "Policy No."],
               policy_app="Existing policy number (if any)", claim=["Claim reference", "Claim no."],
               customer=["Customer number", "Member ID"], street=["Street address", "Address line 1", "House no. and street"],
               city=["Town/City", "City"], postcode=["Postcode", "Postal code"], zipc="ZIP code", country="Country",
               prev="Previous address (if at current address less than 3 years)",
               corr="Correspondence address (if different)", same="as above", phone="Telephone", email="Email",
               contact="Preferred contact method", contact_opts=["Email", "Phone", "Post"],
               incident="Date of incident", start="Cover start date", signed="Date", sig="Signature",
               partner="Second person / partner", partner_dob="Date of birth", block="Please complete in BLOCK CAPITALS",
               code="Form {c} (rev. {m}/{y})", other="Do you hold any other insurance for this risk?", yes="Yes", no="No",
               decl="I declare that the information given on this form is true and complete to the best of my knowledge.",
               fmtnote="Dates: {f}"),
    "de": dict(kinds=[("Schadenmeldung Hausratversicherung", "claim", ["Hausrat", "Wohngebäude", "Glasbruch",
                                                                       "Elementarschäden", "Fahrradschutz", "Rechtsschutz"]),
                      ("Antrag Kfz-Versicherung", "app", ["Teilkasko", "Vollkasko", "Schutzbrief", "Rabattschutz",
                                                          "Fahrerschutz"])],
               ins=["Versicherung AG", "Versicherungsverein a.G."], sec_a="A. ANGABEN ZUM VERSICHERUNGSNEHMER",
               sec_b="B. ANSCHRIFT", sec_c=["C. GEWÜNSCHTER VERSICHERUNGSUMFANG (bitte ankreuzen)"],
               sec_d="D. SONSTIGE ANGABEN",
               surname=["Name", "Nachname"], first=["Vorname"], full=["Vor- und Nachname"], lastfirst="Name, Vorname",
               title="Anrede", titles=["Herr", "Frau"], dob=["Geburtsdatum"],
               policy=["Versicherungsscheinnummer", "Vertragsnummer"], policy_app="Bestehende Vertragsnummer (falls vorhanden)",
               claim=["Schadennummer"], customer=["Kundennummer"], street=["Straße, Hausnummer"], city=["Ort"],
               postcode=["PLZ"], zipc="PLZ", country="Land", prev="Frühere Anschrift (falls weniger als 3 Jahre)",
               corr="Postanschrift (falls abweichend)", same="wie oben", phone="Telefon", email="E-Mail",
               contact="Bevorzugter Kontakt", contact_opts=["E-Mail", "Telefon", "Post"], incident="Schadendatum",
               start="Versicherungsbeginn", signed="Datum", sig="Unterschrift", partner="Mitversicherte Person",
               partner_dob="Geburtsdatum", block="Bitte in Druckbuchstaben ausfüllen", code="Formular {c} ({m}/{y})",
               other="Bestehen weitere Versicherungen?", yes="Ja", no="Nein",
               decl="Ich versichere, dass die Angaben vollständig und richtig sind.", fmtnote="Datumsformat: {f}"),
    "fr": dict(kinds=[("Déclaration de sinistre habitation", "claim", ["Incendie", "Dégâts des eaux", "Vol",
                                                                       "Bris de glace", "Responsabilité civile",
                                                                       "Protection juridique"]),
                      ("Demande d'adhésion assurance santé", "app", ["Dentaire", "Optique", "Hospitalisation",
                                                                     "Médecines douces", "Assistance à l'étranger"])],
               ins=["Assurances SA", "Mutuelle d'Assurance"], sec_a="A. SOUSCRIPTEUR", sec_b="B. ADRESSE",
               sec_d="D. AUTRES RENSEIGNEMENTS",
               sec_c=["C. GARANTIES SOUHAITÉES (cochez les cases)"], surname=["Nom"], first=["Prénom(s)"],
               full=["Nom et prénom"], lastfirst="Nom, prénom", title="Civilité", titles=["M.", "Mme"],
               dob=["Date de naissance"], policy=["N° de contrat", "N° de police"],
               policy_app="N° de contrat existant (le cas échéant)", claim=["N° de sinistre"], customer=["N° de sociétaire"],
               street=["Adresse (n° et rue)"], city=["Ville"], postcode=["Code postal"], zipc="Code postal",
               country="Pays", prev="Adresse précédente (si moins de 3 ans)", corr="Adresse de correspondance (si différente)",
               same="identique", phone="Téléphone", email="E-mail", contact="Contact préféré",
               contact_opts=["E-mail", "Téléphone", "Courrier"], incident="Date du sinistre", start="Date d'effet",
               signed="Fait le", sig="Signature", partner="Conjoint", partner_dob="Date de naissance du conjoint",
               block="Merci d'écrire en MAJUSCULES", code="Formulaire {c} ({m}/{y})",
               other="Êtes-vous assuré auprès d'une autre compagnie ?", yes="Oui", no="Non",
               decl="Je certifie l'exactitude des renseignements ci-dessus.", fmtnote="Format des dates : {f}"),
    "bg": dict(kinds=[("Уведомление за щета – имущество", "claim", ["Пожар", "Наводнение", "Кражба", "Земетресение",
                                                                    "Гражданска отговорност", "Счупване на стъкла"]),
                      ("Предложение-въпросник за застраховка Живот", "app", ["Злополука", "Болест",
                                                                            "Трайна нетрудоспособност",
                                                                            "Медицински разходи"])],
               ins=["Застрахователно дружество АД", "Застраховане ЕАД"], sec_a="I. ДАННИ ЗА ЗАСТРАХОВАНИЯ",
               sec_b="II. АДРЕС", sec_c=["III. ИЗБРАНИ ПОКРИТИЯ (отбележете)"], sec_d="IV. ДРУГА ИНФОРМАЦИЯ",
               surname=["Фамилия"], first=["Име"],
               full=["Трите имена"], lastfirst="", title="", titles=[], dob=["Дата на раждане"],
               policy=["Номер на полица", "Полица №"], policy_app="Номер на съществуваща полица (ако има)",
               claim=["Щета №"], customer=["Клиентски номер"], street=["Адрес (ул., №)"], city=["Град"],
               postcode=["Пощенски код"], zipc="Пощенски код", country="Държава", prev="Предишен адрес",
               corr="Адрес за кореспонденция (ако е различен)", same="същият", phone="Телефон", email="Имейл",
               contact="Предпочитан начин за връзка", contact_opts=["Имейл", "Телефон", "Поща"],
               incident="Дата на събитието", start="Начало на застраховката", signed="Дата", sig="Подпис",
               partner="Второ лице", partner_dob="Дата на раждане", block="Попълва се с печатни букви",
               code="Образец {c} ({m}/{y})", other="Имате ли друга застраховка?", yes="Да", no="Не",
               decl="Декларирам, че посочените данни са верни.", fmtnote="Формат на датите: {f}"),
}
BOXES = [("[X]", "[ ]"), ("[x]", "[ ]"), ("☒", "☐"), ("☑", "☐"), ("(X)", "( )"), ("■", "□"), ("[✔]", "[ ]"),
         ("|X|", "|_|")]


def gen_form(d: Doc):
    rng, loc, lang = d.rng, d.loc, d.lang
    L = FORM[lang]
    title, kind, options = rng.choice(L["kinds"])
    ins = f"{rng.choice(EN_A if lang == 'en' else DE_A if lang == 'de' else FR_A if lang == 'fr' else BG_A)} {rng.choice(L['ins'])}"
    caps = rng.random() < 0.5
    up = (lambda s: s.upper()) if caps else (lambda s: s)
    p = person(rng, loc)
    dob = rand_date(rng, dt.date(1948, 1, 1), dt.date(2004, 12, 28))
    a = address(rng, loc)
    show_country = rng.random() < 0.7
    pol = f"{rng.choice(['HX', 'PL', 'MT', 'TR', 'HP', 'VS'])}-{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}"
    has_pol = kind == "claim" or rng.random() < 0.5
    on, off = rng.choice(BOXES)
    k = rng.choices([0, 1, 2, 3], [0.15, 0.35, 0.3, 0.2])[0]
    ticked = rng.sample(options, k)
    out = [up(ins) if rng.random() < 0.5 else ins, title.upper() if rng.random() < 0.5 else title,
           L["code"].format(c=f"{rng.choice('ABCHKM')}{rng.randint(10, 99)}", m=f"{rng.randint(1, 12):02d}",
                            y=rng.randint(2019, 2024)), noise(L["block"], rng), ""]
    out.append(L["sec_a"])
    if kind == "claim":
        out.append(f"{rng.choice(L['claim'])}: {rng.choice(['CLM', 'S', 'SIN'])}{rng.randint(100000, 999999)}")
    if rng.random() < 0.5:
        out.append(f"{rng.choice(L['customer'])}: {rng.randint(1000000, 9999999)}")
    first, mid, last = p
    style = rng.choice(["split", "lastfirst", "full"] if L["lastfirst"] else ["split", "full"])
    if lang == "bg":
        if style == "split":
            out.append(f"{L['first'][0]}: {up(first)}   Презиме: {up(mid)}   {L['surname'][0]}: {up(last)}")
        else:
            out.append(f"{L['full'][0]}: {up(full(p))}")
    elif style == "split":
        tt = f"{L['title']}: {rng.choice(L['titles'])}    " if rng.random() < 0.5 else ""
        out.append(f"{tt}{rng.choice(L['surname'])}: {up(last)}" + " " * rng.randint(3, 10) +
                   f"{rng.choice(L['first'])}: {up(first)}")
    elif style == "lastfirst":
        out.append(f"{L['lastfirst']}: {up(last)}, {up(first)}")
    else:
        out.append(f"{rng.choice(L['full'])}: {up(first)} {up(last)}")
    dobs = d.date(dob)
    pol_line = (f"{rng.choice(L['policy'])}: {pol}" if kind == "claim" else
                f"{L['policy_app']}: {pol if has_pol else '_' * rng.randint(8, 14)}")
    out += two_col([f"{rng.choice(L['dob'])}: {dobs}"], [pol_line], rng) if rng.random() < 0.5 else [
        f"{rng.choice(L['dob'])}: {dobs}", pol_line]
    if lang == "bg":
        yy, mm = dob.year % 100, dob.month + (40 if dob.year >= 2000 else 0)
        out.append(f"ЕГН: {yy:02d}{mm:02d}{dob.day:02d}{rng.randint(1000, 9999)}")
    out.append(f"{L['phone']}: {phone(rng, loc)}    "
               f"{L['email']}: client{rng.randint(100, 9999)}@example.com")
    out.append("")
    out.append(L["sec_b"])
    pc_label = L["zipc"] if loc == "US" else rng.choice(L["postcode"])
    if rng.random() < 0.5:
        out.append(f"{rng.choice(L['street'])}: {up(a['street'])}")
        out += two_col([f"{rng.choice(L['city'])}: {up(a['city'])}"], [f"{pc_label}: {up(a['postal'])}"], rng)
    else:
        out += [f"{rng.choice(L['street'])}: {up(a['street'])}", f"{pc_label}: {up(a['postal'])}",
                f"{rng.choice(L['city'])}: {up(a['city'])}"]
    country_print = rng.choice(COUNTRY_PRINT[loc])
    if show_country:
        out.append(f"{L['country']}: {up(country_print)}")
    if rng.random() < 0.3:
        b = address(rng, loc)
        while b["city"] == a["city"]:
            b = address(rng, loc)
        out.append(f"{L['prev']}: {up(', '.join(b['lines']))}")
    else:
        out.append(f"{L['prev']}: " + "_" * rng.randint(10, 30))
    if rng.random() < 0.3:
        out.append(f"{L['corr']}: {L['same']}")
    copts = L["contact_opts"]
    cpick = rng.choice(copts)
    out.append(f"{L['contact']}: " + "   ".join(f"{on if o == cpick else off} {o}" for o in copts))
    out.append("")
    out.append(rng.choice(L["sec_c"]))
    cells = [f"{on if o in ticked else off} {o}" for o in options]
    if rng.random() < 0.5:
        out += cells
    else:
        for i in range(0, len(cells), 2):
            out += two_col([cells[i]], [cells[i + 1]] if i + 1 < len(cells) else [], rng, gap=rng.randint(4, 10))
    out += ["", L["sec_d"], f"{L['other']}  {off} {L['yes']}  {on} {L['no']}"]
    out.append("")
    if kind == "claim":
        out.append(f"{L['incident']}: {d.date(rand_date(rng), d.dstyle2)}")
    else:
        out.append(f"{L['start']}: {d.date(rand_date(rng), d.dstyle2)}")
    if rng.random() < 0.3:
        q = person(rng, loc)
        out.append(f"{L['partner']}: {up(full(q))}    {L['partner_dob']}: "
                   f"{d.date(rand_date(rng, dt.date(1950, 1, 1), dt.date(2003, 12, 1)))}")
    out.append(noise(L["decl"], rng))
    out.append(f"{L['sig']}: {up(full(p)) if rng.random() < 0.5 else '__________'}     {L['signed']}: "
               f"{d.date(rand_date(rng))}")
    gold = dict(full_name=full(p), date_of_birth=dob.isoformat(), policy_number=pol if has_pol else None,
                street=a["street"], postal_code=a["postal"], city=a["city"],
                country=COUNTRY_GOLD[loc] if show_country else None, selected_options=ticked)
    d.ev += [dobs] + ([pol] if has_pol else [])
    d.ev_ci = [first, last, a["street"], a["postal"], a["city"]] + ticked
    return out, gold


# ---- bank statements ------------------------------------------------------------------------------------------------
STM = {
    "en": dict(banks=["First Harbor Bank", "Northgate Savings Bank", "Union Crest Bank", "Riverside Building Society"],
               title=["STATEMENT OF ACCOUNT", "Account Statement", "Current Account Statement"],
               holder="Account holder", acct="Account number", sort="Sort code", iban="IBAN",
               period="Statement period", sdate="Statement date", cols5=("Date", "Description", "Paid out", "Paid in", "Balance"),
               cols5b=("Date", "Details", "Debit", "Credit", "Balance"), cols4=("Date", "Description", "Amount", "Balance"),
               open=["Opening balance", "Balance brought forward"], close=["Closing balance", "Balance carried forward"],
               cfwd="Balance carried forward", bfwd="Balance brought forward", avail="Available balance",
               od="Arranged overdraft limit", pending="Pending transactions (not yet included in your balance)",
               tdeb="Total paid out", tcred="Total paid in", page="Page {i} of {n}", cont="continued overleaf",
               rate="Credit interest rate 0.10% AER", to="to", dr="DR", cr="CR",
               desc_d=["CARD PAYMENT TO {m}", "DIRECT DEBIT {m}", "CASH WITHDRAWAL ATM {c}", "STANDING ORDER RENT",
                       "BILL PAYMENT {m}", "MONTHLY ACCOUNT FEE", "CARD PAYMENT {m} {c}"],
               desc_c=["SALARY {m}", "TRANSFER FROM {p}", "REFUND {m}", "INTEREST PAID", "FASTER PAYMENT {p}"],
               merch=["TESCO STORES 2231", "SHELL 4471", "AMAZON MKTPLACE", "BRITISH GAS", "VODAFONE", "COUNCIL TAX",
                      "NETFLIX.COM", "TRAINLINE", "BOOTS 0912", "IKEA", "PRET A MANGER"],
               note="Please check this statement and tell us about any errors within 30 days."),
    "de": dict(banks=["Sparkasse Lindenau", "Volksbank Mittelrhein eG", "Nordbank AG"],
               title=["KONTOAUSZUG", "Kontoauszug"], holder="Kontoinhaber", acct="Kontonummer", sort="BLZ", iban="IBAN",
               period="Zeitraum", sdate="Auszugsdatum", cols5=("Datum", "Buchungstext", "Soll", "Haben", "Saldo"),
               cols5b=("Buchung", "Verwendungszweck", "Lastschrift", "Gutschrift", "Kontostand"),
               cols4=("Datum", "Buchungstext", "Betrag", "Saldo"), open=["Alter Kontostand", "Anfangssaldo"],
               close=["Neuer Kontostand", "Endsaldo"], cfwd="Übertrag", bfwd="Übertrag von Seite {i}",
               avail="Verfügbarer Betrag", od="Eingeräumte Kontoüberziehung",
               pending="Vorgemerkte Umsätze (noch nicht gebucht)", tdeb="Summe Soll", tcred="Summe Haben",
               page="Seite {i} von {n}", cont="Fortsetzung auf der nächsten Seite", rate="Sollzins 11,25 % p.a.",
               to="bis", dr="S", cr="H",
               desc_d=["Kartenzahlung {m}", "Lastschrift {m}", "Bargeldauszahlung GA {c}", "Dauerauftrag Miete",
                       "Überweisung an {p}", "Kontoführungsentgelt"],
               desc_c=["Gehalt {m}", "Gutschrift {p}", "Rückerstattung {m}", "Zinsgutschrift"],
               merch=["REWE MARKT", "ARAL TANKSTELLE", "AMAZON EU", "STADTWERKE", "TELEKOM", "DB VERTRIEB",
                      "DM DROGERIE", "LIDL 4471"],
               note="Bitte prüfen Sie diesen Auszug. Einwendungen sind innerhalb von sechs Wochen zu erheben."),
    "fr": dict(banks=["Banque du Rivage", "Crédit des Lacs", "Caisse Régionale du Val"],
               title=["RELEVÉ DE COMPTE", "Relevé de compte courant"], holder="Titulaire", acct="N° de compte",
               sort="Code banque", iban="IBAN", period="Période", sdate="Date d'arrêté",
               cols5=("Date", "Libellé", "Débit", "Crédit", "Solde"), cols5b=("Date", "Opération", "Débit", "Crédit", "Solde"),
               cols4=("Date", "Libellé", "Montant", "Solde"), open=["Solde précédent", "Ancien solde"],
               close=["Nouveau solde", "Solde au {d}"], cfwd="À reporter", bfwd="Report", avail="Solde disponible",
               od="Autorisation de découvert", pending="Opérations en cours (non comptabilisées)",
               tdeb="Total des débits", tcred="Total des crédits", page="Page {i}/{n}", cont="suite page suivante",
               rate="Taux débiteur 15,90 %", to="au", dr="D", cr="C",
               desc_d=["PAIEMENT CB {m}", "PRÉLÈVEMENT {m}", "RETRAIT DAB {c}", "VIREMENT LOYER", "FRAIS DE TENUE DE COMPTE"],
               desc_c=["VIREMENT SALAIRE {m}", "VIREMENT DE {p}", "REMBOURSEMENT {m}"],
               merch=["CARREFOUR", "TOTALENERGIES", "EDF", "ORANGE", "SNCF", "FNAC", "MONOPRIX"],
               note="Vérifiez ce relevé et signalez-nous toute anomalie dans un délai de 30 jours."),
    "bg": dict(banks=["Банка Витоша АД", "Тракийска Търговска Банка АД"], title=["ИЗВЛЕЧЕНИЕ ПО СМЕТКА"],
               holder="Титуляр", acct="Сметка", sort="BIC", iban="IBAN", period="Период", sdate="Дата на извлечението",
               cols5=("Дата", "Описание", "Дебит", "Кредит", "Салдо"), cols5b=("Дата", "Основание", "Дебит", "Кредит", "Наличност"),
               cols4=("Дата", "Описание", "Сума", "Салдо"), open=["Начално салдо"], close=["Крайно салдо"],
               cfwd="Пренос", bfwd="Пренос от стр. {i}", avail="Разполагаема наличност", od="Разрешен овърдрафт",
               pending="Блокирани суми (неосчетоводени)", tdeb="Общо дебит", tcred="Общо кредит",
               page="Стр. {i} от {n}", cont="продължава на следващата страница", rate="Лихвен процент 0,01%",
               to="до", dr="Д", cr="К",
               desc_d=["Плащане с карта {m}", "Теглене АТМ {c}", "Директен дебит {m}", "Превод към {p}", "Такса обслужване"],
               desc_c=["Заплата {m}", "Превод от {p}", "Възстановена сума {m}"],
               merch=["КАУФЛАНД", "ЛУКОЙЛ", "ЕВН БЪЛГАРИЯ", "А1", "СОФИЙСКА ВОДА", "БИЛА"],
               note="Моля, проверете извлечението и ни уведомете за несъответствия в 30-дневен срок."),
}


def gen_statement(d: Doc):
    rng, loc, lang = d.rng, d.loc, d.lang
    L = STM[lang]
    bank = rng.choice(L["banks"])
    holder = full(person(rng, loc)) if rng.random() < 0.7 else company(rng, loc)
    start = rand_date(rng, dt.date(2024, 1, 1), dt.date(2026, 8, 1)).replace(day=1)
    end = (start.replace(day=28) + dt.timedelta(days=4)).replace(day=1) - dt.timedelta(days=1)
    opening = D(rng.uniform(300, 6000)) if rng.random() < 0.9 else D(rng.uniform(-900, -50))
    n = rng.randint(6, 14)
    days = sorted(start + dt.timedelta(days=rng.randrange((end - start).days + 1)) for _ in range(n))
    txs, bal = [], opening
    city = rng.choice([c[0] for c in UK_CITY] if lang == "en" else ["BERLIN", "PARIS", "SOFIA", "LYON"])
    # kinds: (index into desc_d / desc_c, low, high, credit); desc_d = card, direct debit, ATM, rent, bill, fee, card;
    # desc_c = salary, transfer in, refund, interest, transfer in
    kinds = ([(0, 3, 180, False)] * 4 + [(1, 15, 260, False)] * 2 + [(2, 20, 300, False), (4, 20, 220, False),
             (len(L["desc_d"]) - 1, 3, 120, False), (1, 15, 260, False)] + [(1, 10, 600, True), (2, 5, 150, True)])
    kinds_once = [(3, 450, 1650, False), (0, 1, 1, True), (5, 1, 15, False)]       # rent, salary, account fee
    picks = [rng.choice(kinds) for _ in range(n)]
    for k in kinds_once:
        if rng.random() < 0.7:
            picks[rng.randrange(n)] = k
    if not any(not c for *_, c in picks):
        picks[0] = kinds[0]
    for t, (ki, lo, hi, credit) in zip(days, picks):
        if credit and lo == 1 and hi == 1:                 # salary from the employer
            desc = L["desc_c"][0].format(m=company(rng, loc).upper())
            amt = D(rng.uniform(1500, 4200))
        else:
            src = L["desc_c"] if credit else L["desc_d"]
            desc = src[min(ki, len(src) - 1)].format(m=rng.choice(L["merch"]), c=city.upper(),
                                                     p=full(person(rng, loc)).upper())
            amt = D(rng.uniform(lo, hi))
            if not credit and ki == 2:                     # cash machines pay round sums
                amt = D(round(float(amt), -1) or 20)
        bal = bal + amt if credit else bal - amt
        txs.append((t, desc, amt, credit, bal))
    closing = bal
    debits = [a for _, _, a, c, _ in txs if not c]
    if not debits:                                       # at least one debit
        t, desc, amt, _, _ = txs[0]
        txs[0] = (t, desc, amt, False, None)
        bal = opening
        txs = [(t, ds, a, c, None) for t, ds, a, c, _ in txs]
        out2 = []
        for t, ds, a, c, _ in txs:
            bal = bal + a if c else bal - a
            out2.append((t, ds, a, c, bal))
        txs, closing = out2, bal
        debits = [a for _, _, a, c, _ in txs if not c]
    layout = rng.choice(["five", "five", "signed", "drcr"])
    show_close = rng.random() < 0.9
    pages = 2 if n >= 10 and rng.random() < 0.6 else 1
    acct = f"{rng.randint(10000000, 99999999)}"
    out = [bank.upper() if rng.random() < 0.5 else bank, rng.choice(L["title"]), ""]
    left = [f"{L['holder']}: {holder}", f"{L['acct']}: {acct}"]
    right = [f"{L['period']}: {d.date(start)} {L['to']} {d.date(end)}", f"{L['sdate']}: {d.date(end + dt.timedelta(days=rng.randint(1, 4)))}"]
    out += two_col(left, right, rng) if rng.random() < 0.5 else left + right
    if rng.random() < 0.5:
        out.append(f"{L['iban']}: {bank_line(rng, 'DE' if lang == 'de' else 'FR' if lang == 'fr' else 'BG' if lang == 'bg' else 'IE').split('IBAN')[-1].strip(': ').split('  BIC')[0]}")
    od = D(rng.choice([500, 1000, 1500, 2500]))
    if rng.random() < 0.5:
        out.append(f"{L['od']}: {d.amt(od)}")
    out.append("")

    def row(t, desc, amt, credit, b):
        ds = d.date(t, "d Mon Y" if lang == "en" and d.dstyle == "dd-Mon-Y" else None)
        if layout == "five":
            return [ds, desc, "" if credit else d.amt(amt), d.amt(amt) if credit else "", d.amt(b)]
        if layout == "signed":
            s = d.amt(amt) if credit else d.amt(-amt)
            return [ds, desc, ("+" + s) if credit and rng.random() < 0.5 else s, d.amt(b)]
        return [ds, desc, f"{d.amt(amt)} {L['cr'] if credit else L['dr']}", d.amt(b)]

    cols = (rng.choice([L["cols5"], L["cols5b"]]) if layout == "five" else L["cols4"])
    ob = rng.choice(L["open"])
    rows = [list(cols), [d.date(start), ob] + ([""] * (len(cols) - 3)) + [d.amt(opening)]]
    split_at = n // 2 + rng.randint(-1, 1) if pages == 2 else n
    lines = []
    for i, tx in enumerate(txs):
        r = row(*tx)
        if layout == "five" and rng.random() < 0.5:          # OCR drops the empty cell: the amount drifts left
            r = [c for c in r if c != ""]
        lines.append(r)
        if i + 1 == split_at and pages == 2:
            b = tx[4]
            lines.append(["", L["cfwd"], d.amt(b)])
            lines.append(["PAGE"])
            lines.append(["", L["bfwd"].format(i=1), d.amt(b)])
    if show_close:
        lines.append([d.date(end), rng.choice(L["close"]).format(d=d.date(end)), d.amt(closing)])
    for r in lines:
        if r == ["PAGE"]:
            rows.append(["PAGE"])
            continue
        rows.append(r)
    # render: wrap long descriptions, page break block
    widths = [12, 30] + [12] * (len(cols) - 2)
    rendered, buf = [], []
    for r in rows:
        if r == ["PAGE"]:
            rendered += table(buf, rng, widths, jitter=3, left=2)
            buf = []
            rendered += ["", noise(L["cont"], rng), L["page"].format(i=1, n=2), "", bank, L["page"].format(i=2, n=2),
                         " ".join(cols), ""]
            continue
        if len(r) > 1 and len(r[1]) > 28 and rng.random() < 0.5:
            cut = r[1].rfind(" ", 0, 28)
            if cut > 0:
                buf.append([r[0], r[1][:cut]] + r[2:])
                buf.append(["", "  " + r[1][cut + 1:]])
                continue
        buf.append(r)
    rendered += table(buf, rng, widths, jitter=3, left=2)
    out += rendered
    out.append("")
    if not show_close or rng.random() < 0.4:
        out.append(f"{L['tdeb']}: {d.amt(sum(debits))}    {L['tcred']}: {d.amt(sum(a for _, _, a, c, _ in txs if c))}")
    if rng.random() < 0.4:
        out.append(f"{L['avail']}: {d.amt(closing + od)}")
    if rng.random() < 0.35:
        out.append(L["pending"] + ":")
        for _ in range(rng.randint(1, 2)):
            pdt = end + dt.timedelta(days=rng.randint(0, 2))
            pa = D(rng.uniform(float(max(debits)) * 0.8, float(max(debits)) * 1.6))
            out.append(f"  {d.date(pdt)}  {rng.choice(L['desc_d']).format(m=rng.choice(L['merch']), c=city.upper(), p='X')}  {d.amt(-pa)}")
    out.append(noise(L["rate"], rng, 0.03))
    out.append(noise(L["note"], rng))
    if pages == 1:
        out.append(L["page"].format(i=1, n=1))
    gold = dict(opening_balance=float(opening), closing_balance=float(closing), transaction_count=n,
                largest_debit=float(max(debits)))
    d.ev += [d.amt(opening), d.amt(closing), d.amt(max(debits))]
    return out, gold


# ---- contracts and letters ------------------------------------------------------------------------------------------
LAW_EN = {"England and Wales": "England and Wales", "Scotland": "Scotland", "Ireland": "Ireland",
          "Germany": "the Federal Republic of Germany", "Austria": "the Republic of Austria", "France": "France",
          "Belgium": "Belgium", "Bulgaria": "the Republic of Bulgaria", "Netherlands": "the Netherlands",
          "Switzerland": "Switzerland", "New York": "the State of New York", "Delaware": "the State of Delaware",
          "California": "the State of California", "Texas": "the State of Texas"}
LAW_DE = {"Germany": "Es gilt das Recht der Bundesrepublik Deutschland unter Ausschluss des UN-Kaufrechts.",
          "Austria": "Dieser Vertrag unterliegt österreichischem Recht.",
          "Switzerland": "Dieser Vertrag untersteht schweizerischem Recht."}
LAW_FR = {"France": "Le présent contrat est régi par le droit français.",
          "Belgium": "Le présent contrat est soumis au droit belge.",
          "Switzerland": "Le présent contrat est régi par le droit suisse."}
LAW_BG = {"Bulgaria": "За неуредените в настоящия договор въпроси се прилага българското законодателство."}
UNIT_EN = {"days": "days", "weeks": "weeks", "months": "months"}
SERV = {"en": ["IT support and maintenance services", "the supply of packaging materials", "facility cleaning services",
               "payroll processing services", "marketing consultancy services", "the lease of office equipment",
               "translation and localisation services", "logistics and warehousing services"],
        "de": ["IT-Wartungsleistungen", "die Lieferung von Verpackungsmaterial", "Gebäudereinigung",
               "Lohnbuchhaltung", "Marketingberatung"],
        "fr": ["des prestations de maintenance informatique", "la fourniture d'emballages", "le nettoyage des locaux",
               "des prestations de conseil en marketing"],
        "bg": ["услуги по поддръжка на информационни системи", "доставка на опаковъчни материали",
               "почистване на офис помещения", "счетоводни услуги"]}


def notice_text(lang, v, unit, rng):
    if lang == "en":
        u = UNIT_EN[unit] if v != 1 else unit[:-1]
        w = num_word(v, "en")
        return rng.choice([f"not less than {w} ({v}) {u}' prior written notice", f"{v} {u}' written notice",
                           f"at least {w} {u}' notice in writing"])
    if lang == "de":
        u = {"days": "Tagen", "weeks": "Wochen", "months": "Monaten" if v != 1 else "Monat"}[unit]
        return f"einer Frist von {num_word(v, 'de') if v in (1, 2, 3, 4, 6) else v} {u}"
    if lang == "fr":
        u = {"days": "jours", "weeks": "semaines", "months": "mois"}[unit]
        return f"un préavis de {num_word(v, 'fr')} ({v}) {u}"
    if unit == "months":
        return {1: "едномесечно", 2: "двумесечно", 3: "тримесечно", 6: "шестмесечно"}[v] + " писмено предизвестие"
    return f"{v}-дневно писмено предизвестие"


def inc(rng, loc):
    """how a party's incorporation is described (a place that need not be the governing law)"""
    if loc == "US":
        return f"a {rng.choice(['Delaware', 'Nevada', 'New York', 'Texas', 'California'])} corporation"
    if loc == "UK":
        where = rng.choice(["England and Wales", "England and Wales", "Scotland"])
        return f"a company incorporated in {where} (company no. {'SC' if where == 'Scotland' else '0'}{rng.randint(100000, 999999)})"
    return f"a company incorporated in {COUNTRY_GOLD[loc]} (registered no. {rng.randint(100000, 999999)})"


def gen_contract(d: Doc):
    rng, loc, lang = d.rng, d.loc, d.lang
    p1 = company(rng, loc)
    loc2 = loc if rng.random() < 0.7 else rng.choice(["UK", "US", "IE"] if lang == "en" else [loc])
    p2 = company(rng, loc2, avoid=(p1,))
    a1, a2 = address(rng, loc), address(rng, loc2)
    eff = rand_date(rng)
    sig1 = eff - dt.timedelta(days=rng.randint(3, 40))
    sig2 = sig1 + dt.timedelta(days=rng.randint(0, 9))
    on_sig = rng.random() < 0.15                          # takes effect on the (last) signature
    if on_sig:
        eff = max(sig1, sig2)
    has_notice = rng.random() < 0.85
    unit = rng.choice(["days", "months", "months", "weeks"] if lang in ("en", "de") else ["days", "months", "months"])
    v = {"days": rng.choice([14, 30, 60, 90]), "weeks": rng.choice([2, 4, 6]), "months": rng.choice([1, 2, 3, 6])}[unit]
    if lang == "bg" and unit == "days":
        v = rng.choice([14, 30, 60])
    laws = {"en": list(LAW_EN), "de": list(LAW_DE), "fr": list(LAW_FR), "bg": list(LAW_BG)}[lang]
    if lang == "en":
        law = {"US": rng.choice(["New York", "Delaware", "California", "Texas"]),
               "UK": rng.choice(["England and Wales", "England and Wales", "Scotland"]),
               "IE": "Ireland"}[loc] if rng.random() < 0.75 else rng.choice(laws)
    else:
        law = rng.choice(laws) if lang != "bg" else "Bulgaria"
    has_law = rng.random() < 0.9
    cure = rng.choice([14, 21, 30])
    pay = rng.choice([14, 30, 45, 60])
    serv = rng.choice(SERV[lang])
    term = rng.choice([12, 24, 36])
    fee = D(rng.choice([1200, 2500, 4800, 7500, 12000]) * rng.choice([1, 1.5, 2]))
    out = []
    letter = lang == "en" and rng.random() < 0.35
    if lang == "en" and letter:
        signer, csign = full(person(rng, loc)), full(person(rng, loc2))
        out += [p1.upper(), *a1["lines"], "", d.date(sig1), "", "Private and confidential", p2, *a2["lines"], "",
                f"Dear {csign.split()[0]},", "", f"Re: Engagement letter - {serv}", "",
                noise(f"We are pleased to confirm the terms on which {p1} (\"we\", \"us\") will provide {serv} to "
                      f"{p2} (\"you\"). This letter and the attached terms of business form the agreement between us.", rng, 0.02)]
        if on_sig:
            out.append(noise("This engagement takes effect on the date on which you countersign this letter.", rng, 0.02))
        else:
            out.append(noise("This engagement will commence on ", rng, 0.02) + d.date(eff) + noise(
                f" and will continue for an initial period of {term} months.", rng, 0.02))
        out.append(noise(f"Our fees are {d.amt(fee, sym=True)} per month, invoiced monthly in arrears and payable "
                         f"within {pay} days of the invoice date.", rng, 0.02))
        if has_notice:
            out.append(noise("Either of us may terminate this engagement at any time, without giving a reason, by giving ", rng, 0.02)
                       + notice_text("en", v, unit, rng) + ".")
        else:
            out.append(noise(f"This engagement may not be terminated before the end of the initial period except for "
                             f"material breach that is not remedied within {cure} days of written notice.", rng, 0.02))
        out.append(noise(f"We will review our fees annually and give you {rng.choice([30, 60])} days' notice of any change.", rng, 0.02))
        out.append(noise("We will keep all information you give us confidential, save as required by law or regulation.", rng))
        if has_law:
            out.append(f"This letter is governed by the laws of {LAW_EN[law]}" + (
                "." if rng.random() < 0.6 else f", and the courts of {rng.choice(['London', 'Edinburgh', 'Dublin', 'New York', 'Wilmington'])} shall have jurisdiction."))
        out += ["", "Yours sincerely,", "", signer, f"Partner, for and on behalf of {p1}", "",
                "Agreed and accepted:", f"for and on behalf of {p2}", f"Name: {csign}", f"Date: {d.date(sig2)}"]
    elif lang == "en":
        title = rng.choice(["SERVICES AGREEMENT", "MASTER SERVICES AGREEMENT", "SUPPLY AGREEMENT",
                            "CONSULTANCY AGREEMENT", "EQUIPMENT LEASE AGREEMENT"])
        out += [title, "", f"THIS AGREEMENT is dated {d.date(sig1)}" if rng.random() < 0.5 else "THIS AGREEMENT is made", "",
                "BETWEEN:", f"(1) {p1.upper() if rng.random() < 0.5 else p1}, {inc(rng, loc)}, whose registered "
                f"office is at {', '.join(a1['lines'])} (the \"Supplier\"); and",
                f"(2) {p2.upper() if rng.random() < 0.5 else p2}, {inc(rng, loc2)}, whose registered office is at "
                f"{', '.join(a2['lines'])} (the \"Customer\").", "",
                "BACKGROUND", noise(f"(A) The Supplier carries on the business of {serv}. (B) The Customer wishes to "
                                    f"obtain those services on the terms of this Agreement.", rng), "",
                "1. COMMENCEMENT AND TERM"]
        if on_sig:
            out.append(noise("1.1 This Agreement shall take effect on the date on which it has been signed by both parties "
                             "(the \"Effective Date\") and shall continue for an initial term of ", rng, 0.02) + f"{term} months.")
        else:
            out.append("1.1 " + rng.choice(["This Agreement shall take effect on ", "This Agreement commences on ",
                                            "The Services shall commence on "]) + d.date(eff) +
                       noise(f" (the \"Effective Date\") and shall continue for an initial term of {term} months, "
                             "after which it shall renew automatically for successive periods of 12 months.", rng, 0.02))
        out += ["", "2. CHARGES AND PAYMENT",
                noise(f"2.1 The Customer shall pay the Supplier {d.amt(fee, sym=True)} per month. 2.2 Invoices are "
                      f"payable within {pay} days of receipt. 2.3 Late payments bear interest at 4% per annum above "
                      "the base rate. 2.4 The Supplier may increase the charges once a year by giving the Customer "
                      f"{rng.choice([30, 60, 90])} days' written notice.", rng, 0.03), "",
                "3. TERMINATION"]
        if has_notice:
            out.append("3.1 Either party may terminate this Agreement for convenience by giving " +
                       notice_text("en", v, unit, rng) + " to the other party.")
        else:
            out.append(noise("3.1 Neither party may terminate this Agreement for convenience during the initial term "
                             "or any renewal term.", rng, 0.02))
        out.append(noise(f"3.2 Either party may terminate this Agreement immediately by written notice if the other "
                         f"party commits a material breach and fails to remedy it within {cure} days of being "
                         "notified of the breach. 3.3 Either party may terminate this Agreement immediately if the "
                         "other party becomes insolvent.", rng, 0.03))
        out += ["", "4. CONFIDENTIALITY", noise("4.1 Each party shall keep the other party's confidential information "
                                                "confidential for the term of this Agreement and five (5) years after "
                                                "its termination.", rng), ""]
        extra = [("DATA PROTECTION", "Each party shall comply with applicable data protection law. The Supplier shall "
                  "notify the Customer of any personal data breach without undue delay and in any event within 72 "
                  "hours of becoming aware of it."),
                 ("FORCE MAJEURE", "Neither party shall be liable for any delay caused by events beyond its reasonable "
                  "control. If such an event continues for more than 60 days, either party may terminate this "
                  "Agreement by giving 14 days' written notice."),
                 ("INSURANCE", f"The Supplier shall maintain public liability insurance of not less than "
                  f"{d.amt(D(rng.choice([1, 2, 5])) * 1000000, sym=True)} for each claim during the term and for "
                  "six (6) years after it."),
                 ("ASSIGNMENT", "Neither party may assign or subcontract its rights under this Agreement without the "
                  "prior written consent of the other party, such consent not to be unreasonably withheld; the "
                  f"Supplier may subcontract to its affiliate {company(rng, loc, avoid=(p1, p2))}."),
                 ("ENTIRE AGREEMENT", "This Agreement constitutes the entire agreement between the parties and "
                  "supersedes all previous agreements, promises and understandings between them, whether written "
                  "or oral, relating to its subject matter."),
                 ("VARIATION", "No variation of this Agreement shall be effective unless it is in writing and signed "
                  "by the parties (or their authorised representatives)."),
                 ("COUNTERPARTS", "This Agreement may be executed in any number of counterparts, each of which when "
                  "executed shall constitute a duplicate original.")]
        out += ["5. LIABILITY", noise(f"5.1 Each party's total liability under this Agreement shall not exceed "
                                      f"{d.amt(fee * 12, sym=True)} in any contract year.", rng, 0.03), "",
                "6. NOTICES", noise(f"6.1 Notices shall be sent to the addresses above, with a copy (which shall not "
                                    f"constitute notice) to {rng.choice(EN_A)} & {rng.choice(LAST['US'])} LLP.", rng, 0.03), ""]
        k = 7
        for h, t in rng.sample(extra, rng.randint(3, 6)):
            out += [f"{k}. {h}", noise(f"{k}.1 " + t, rng, 0.04), ""]
            k += 1
        if has_law:
            out.append(f"{k}. GOVERNING LAW")
            out.append(f"{k}.1 This Agreement and any dispute arising from it shall be governed by the laws of {LAW_EN[law]}.")
            if rng.random() < 0.4:
                out.append(noise(f"{k}.2 Any dispute shall be finally settled by arbitration under the ICC Rules, seated in "
                                 f"{rng.choice(['Geneva', 'Paris', 'Stockholm', 'Singapore'])}.", rng, 0.02))
        out += ["", "Signed by the parties:", "",
                f"For and on behalf of {p1}" + " " * rng.randint(4, 12) + f"For and on behalf of {p2}",
                f"Name: {full(person(rng, loc))}" + " " * rng.randint(8, 16) + f"Name: {full(person(rng, loc2))}",
                f"Date: {d.date(sig1)}" + " " * rng.randint(12, 22) + f"Date: {d.date(sig2)}"]
    elif lang == "de":
        out += [rng.choice(["DIENSTLEISTUNGSVERTRAG", "RAHMENVERTRAG", "WARTUNGSVERTRAG"]), "", "zwischen", "",
                p1, ", ".join(a1["lines"]), "- nachfolgend \"Auftragnehmer\" -", "", "und", "", p2, ", ".join(a2["lines"]),
                "- nachfolgend \"Auftraggeber\" -", "", "§ 1 Vertragsgegenstand",
                noise(f"Der Auftragnehmer erbringt für den Auftraggeber {serv}.", rng), "", "§ 2 Vertragsbeginn und Laufzeit"]
        if on_sig:
            out.append(noise("Der Vertrag tritt mit Unterzeichnung durch beide Parteien in Kraft und läuft auf unbestimmte Zeit.", rng, 0.02))
        else:
            out.append(rng.choice(["Der Vertrag tritt am ", "Der Vertrag beginnt am "]) + d.date(eff) +
                       rng.choice([" in Kraft und läuft auf unbestimmte Zeit.", " und wird auf unbestimmte Zeit geschlossen."]))
        out += ["", "§ 3 Vergütung", noise(f"Die monatliche Vergütung beträgt {d.amt(fee, sym=True)} zzgl. MwSt. "
                                         f"Rechnungen sind innerhalb von {pay} Tagen zahlbar.", rng, 0.03), "",
                "§ 4 Kündigung"]
        if has_notice:
            out.append(f"(1) Der Vertrag kann von jeder Partei mit {notice_text('de', v, unit, rng)} zum Monatsende "
                       "ordentlich gekündigt werden.")
        else:
            out.append(noise(f"(1) Eine ordentliche Kündigung ist während einer Laufzeit von {term} Monaten ausgeschlossen.", rng, 0.02))
        out.append(noise("(2) Das Recht zur außerordentlichen Kündigung aus wichtigem Grund bleibt unberührt. (3) Die "
                         "Kündigung bedarf der Schriftform.", rng))
        out += ["", "§ 5 Haftung", noise(f"Die Haftung ist auf {d.amt(fee * 12, sym=True)} je Vertragsjahr begrenzt.", rng, 0.03), "",
                "§ 6 Schlussbestimmungen"]
        if has_law:
            out.append(LAW_DE[law] + (f" Gerichtsstand ist {rng.choice(['München', 'Hamburg', 'Wien', 'Zürich'])}."
                                      if rng.random() < 0.6 else ""))
        out.append(noise("Sollte eine Bestimmung unwirksam sein, bleibt der Vertrag im Übrigen wirksam.", rng))
        out += ["", f"{a1['city']}, den {d.date(sig1)}" + " " * rng.randint(6, 14) + f"{a2['city']}, den {d.date(sig2)}",
                "____________________" + " " * 10 + "____________________", "Auftragnehmer" + " " * 17 + "Auftraggeber"]
    elif lang == "fr":
        out += [rng.choice(["CONTRAT DE PRESTATION DE SERVICES", "CONTRAT-CADRE", "CONTRAT DE MAINTENANCE"]), "",
                "ENTRE LES SOUSSIGNÉS :", "", f"{p1}, dont le siège social est situé {', '.join(a1['lines'])}, "
                "ci-après dénommée « le Prestataire »,", "", "ET", "",
                f"{p2}, dont le siège social est situé {', '.join(a2['lines'])}, ci-après dénommée « le Client ».", "",
                "Article 1 - Objet", noise(f"Le Prestataire assure pour le compte du Client {serv}.", rng), "",
                "Article 2 - Date d'effet et durée"]
        if on_sig:
            out.append(noise("Le présent contrat entre en vigueur à la date de sa signature par la dernière des parties.", rng, 0.02))
        else:
            out.append(rng.choice(["Le présent contrat prend effet le ", "Le présent contrat entre en vigueur le "]) +
                       d.date(eff) + " pour une durée indéterminée.")
        out += ["", "Article 3 - Prix", noise(f"Le prix mensuel est fixé à {d.amt(fee, sym=True)} HT. Les factures sont "
                                            f"payables à {pay} jours.", rng, 0.03), "", "Article 4 - Résiliation"]
        if has_notice:
            out.append("Chacune des parties peut résilier le présent contrat à tout moment, moyennant " +
                       notice_text("fr", v, unit, rng) + ", par lettre recommandée avec accusé de réception.")
        else:
            out.append(noise(f"Le contrat ne peut être résilié avant l'expiration d'une période de {term} mois.", rng, 0.02))
        out.append(noise(f"En cas de manquement grave, le contrat pourra être résilié de plein droit {cure} jours après "
                         "une mise en demeure restée sans effet.", rng, 0.03))
        out += ["", "Article 5 - Droit applicable"]
        if has_law:
            out.append(LAW_FR[law] + (f" Tout litige sera soumis aux tribunaux compétents de {rng.choice(['Lyon', 'Paris', 'Bruxelles', 'Genève'])}."
                                      if rng.random() < 0.6 else ""))
        else:
            out.append(noise("Les parties s'efforceront de régler à l'amiable tout différend.", rng))
        out += ["", f"Fait à {a1['city']}, le {d.date(sig1)}, en deux exemplaires.", "",
                "Pour le Prestataire" + " " * 14 + "Pour le Client", f"Signé le {d.date(sig1)}" + " " * 12 + f"Signé le {d.date(sig2)}"]
    else:
        out += ["ДОГОВОР", f"за {serv}", "", f"Днес, {d.date(sig2)}, в гр. {a1['city']}, между:", "",
                f"1. {p1}, със седалище и адрес на управление {', '.join(a1['lines'])}, ЕИК {rng.randint(100000000, 999999999)}, "
                f"представлявано от {full(person(rng, loc))} – управител, наричано по-долу ИЗПЪЛНИТЕЛ, и", "",
                f"2. {p2}, със седалище и адрес на управление {', '.join(a2['lines'])}, ЕИК {rng.randint(100000000, 999999999)}, "
                f"наричано по-долу ВЪЗЛОЖИТЕЛ,", "", "се сключи настоящият договор за следното:", "",
                "I. ПРЕДМЕТ", noise(f"Чл. 1. ИЗПЪЛНИТЕЛЯТ се задължава да извършва {serv}.", rng), "", "II. СРОК"]
        if on_sig:
            eff = sig2
            out.append(noise("Чл. 2. Договорът влиза в сила от датата на подписването му и е безсрочен.", rng, 0.02))
        else:
            out.append(f"Чл. 2. Договорът влиза в сила от {d.date(eff)} и е безсрочен.")
        out += ["", "III. ЦЕНА", noise(f"Чл. 3. Месечното възнаграждение е {d.amt(fee, sym=True)} без ДДС, платимо в срок "
                                       f"до {pay} дни от издаване на фактура.", rng, 0.03), "", "IV. ПРЕКРАТЯВАНЕ"]
        if has_notice:
            out.append(f"Чл. 4. Всяка от страните може да прекрати договора с {notice_text('bg', v, unit, rng)}.")
        else:
            out.append(noise(f"Чл. 4. Договорът не може да бъде прекратен едностранно преди изтичане на {term} месеца.", rng, 0.02))
        out.append(noise(f"Чл. 5. При неизпълнение изправната страна може да развали договора с {cure}-дневен срок за изпълнение.", rng, 0.03))
        out += ["", "V. ЗАКЛЮЧИТЕЛНИ РАЗПОРЕДБИ"]
        if has_law:
            out.append("Чл. 6. " + LAW_BG[law] + (f" Споровете се решават от компетентния съд в гр. {rng.choice(['София', 'Пловдив', 'Варна'])}."
                                                    if rng.random() < 0.6 else ""))
        out += ["", "ИЗПЪЛНИТЕЛ: ............" + " " * 10 + "ВЪЗЛОЖИТЕЛ: ............"]
    gold = dict(parties=[p1, p2], effective_date=eff.isoformat(),
                notice_period=dict(value=v, unit=unit) if has_notice else None,
                governing_law=law if has_law else None)
    d.ev_ci = [p1, p2]
    d.ev += [d.date(eff, record=False)] if not on_sig else []
    return out, gold


GEN = {"invoice": gen_invoice, "receipt": gen_receipt, "form": gen_form, "statement": gen_statement,
       "contract": gen_contract}
NOTE = {"en": "All dates are shown as {f}.", "de": "Datumsformat: {f}", "fr": "Format des dates : {f}",
        "bg": "Формат на датите: {f}"}


def build(doc_type: str, i: int) -> dict:
    rng = random.Random(f"{SEED}-{doc_type}-{i}")
    lang = PLAN[i]
    loc = {"de": "DE", "fr": "FR", "bg": "BG"}.get(lang) or rng.choices(["US", "UK", "IE"], [0.45, 0.4, 0.15])[0]
    d = Doc(rng, loc)
    d.ev_ci = []
    lines, gold = GEN[doc_type](d)
    note = d.date_note()
    if note:
        k = next((j for j, x in enumerate(lines) if j > 2 and x == ""), len(lines))
        lines.insert(k, NOTE[d.lang].format(f=note))
    text = "\n".join(x.rstrip() for x in lines)
    miss = [e for e in d.ev if e not in text] + [e for e in d.ev_ci if e.casefold() not in text.casefold()]
    assert not miss, (doc_type, i, miss)
    return dict(id=f"{doc_type}-{i:02d}", type=doc_type, lang=lang, loc=loc, text=text, gold=gold,
                date_note=note is not None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "set", "docs.jsonl"))
    ap.add_argument("--show", default="")
    a = ap.parse_args()
    docs = [build(t, i) for t in TYPES for i in range(len(PLAN))]
    if a.show:
        t, n = a.show.split(":")
        doc = next(x for x in docs if x["id"] == f"{t}-{int(n):02d}")
        print(doc["text"])
        print(json.dumps(doc["gold"], ensure_ascii=False, indent=1))
        return
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for x in docs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    h = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    nf = sum(len(x["gold"]) for x in docs)
    print(f"{len(docs)} documents, {nf} fields, {sum(x['date_note'] for x in docs)} with a date-format note -> {a.out}")
    print(f"sha256 {h}")


if __name__ == "__main__":
    main()
