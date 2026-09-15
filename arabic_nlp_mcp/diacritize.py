"""Arabic diacritization (tashkeel restoration).

Full automatic diacritization is a genuinely hard NLP problem - it needs
either a large morphological disambiguation system (CAMeL Tools, MADAMIRA)
or a trained sequence model, and both need dependencies or model weights
this sandbox can't reach while building the package (see sentiment.py for
the same network constraint). Shipping a fake or silently-wrong
diacritizer would be worse than not having one.

So this ships as what it honestly is: a dictionary lookup over ~150
common, high-frequency Arabic words (function words, pronouns, common
verbs/nouns) with their correct diacritics, applied word-by-word. Anything
outside the dictionary is left undiacritized rather than guessed. The
result reports per-word coverage so callers always know how much of the
output is real versus untouched - a 30%-covered casual sentence and a
90%-covered formal one are very different results, and this tool doesn't
blur that line.
"""

from __future__ import annotations

import re
import unicodedata

from .common import validate_text
from .models import DiacritizeResult, WordReport

# word (undiacritized) -> diacritized form. Deliberately small and curated;
# extend via CONTRIBUTING.md rather than trying to autogenerate this list.
_DICTIONARY = {
    "من": "مِنْ",
    "الى": "إِلَى",
    "إلى": "إِلَى",
    "في": "فِي",
    "على": "عَلَى",
    "عن": "عَنْ",
    "مع": "مَعَ",
    "هذا": "هَذَا",
    "هذه": "هَذِهِ",
    "ذلك": "ذَلِكَ",
    "تلك": "تِلْكَ",
    "هو": "هُوَ",
    "هي": "هِيَ",
    "هم": "هُمْ",
    "انت": "أَنْتَ",
    "أنت": "أَنْتَ",
    "انا": "أَنَا",
    "أنا": "أَنَا",
    "نحن": "نَحْنُ",
    "الذي": "الَّذِي",
    "التي": "الَّتِي",
    "الذين": "الَّذِينَ",
    "لا": "لَا",
    "لم": "لَمْ",
    "لن": "لَنْ",
    "ان": "إِنَّ",
    "إن": "إِنَّ",
    "كان": "كَانَ",
    "كانت": "كَانَتْ",
    "يكون": "يَكُونُ",
    "قال": "قَالَ",
    "قالت": "قَالَتْ",
    "جاء": "جَاءَ",
    "ذهب": "ذَهَبَ",
    "علم": "عِلْمٌ",
    "عمل": "عَمَلٌ",
    "بيت": "بَيْتٌ",
    "كتاب": "كِتَابٌ",
    "مدرسة": "مَدْرَسَةٌ",
    "جامعة": "جَامِعَةٌ",
    "طالب": "طَالِبٌ",
    "معلم": "مُعَلِّمٌ",
    "مهندس": "مُهَنْدِسٌ",
    "طبيب": "طَبِيبٌ",
    "ماء": "مَاءٌ",
    "شمس": "شَمْسٌ",
    "قمر": "قَمَرٌ",
    "نور": "نُورٌ",
    "علي": "عَلِيٌّ",
    "محمد": "مُحَمَّدٌ",
    "احمد": "أَحْمَدُ",
    "خير": "خَيْرٌ",
    "شر": "شَرٌّ",
    "حق": "حَقٌّ",
    "صدق": "صِدْقٌ",
    "كذب": "كَذِبٌ",
    "جميل": "جَمِيلٌ",
    "قبيح": "قَبِيحٌ",
    "كبير": "كَبِيرٌ",
    "صغير": "صَغِيرٌ",
    "طويل": "طَوِيلٌ",
    "قصير": "قَصِيرٌ",
    "جديد": "جَدِيدٌ",
    "قديم": "قَدِيمٌ",
    "اليوم": "الْيَوْمَ",
    "غدا": "غَدًا",
    "امس": "أَمْسِ",
    "الان": "الْآنَ",
    "دائما": "دَائِمًا",
    "ابدا": "أَبَدًا",
    "احيانا": "أَحْيَانًا",
    "سلام": "سَلَامٌ",
    "شكرا": "شُكْرًا",
    "عفوا": "عَفْوًا",
    "نعم": "نَعَمْ",
    "لماذا": "لِمَاذَا",
    "كيف": "كَيْفَ",
    "متى": "مَتَى",
    "اين": "أَيْنَ",
    "من هو": "مَنْ هُوَ",
    "ماذا": "مَاذَا",
    "كل": "كُلُّ",
    "بعض": "بَعْضُ",
    "غير": "غَيْرُ",
    "دون": "دُونَ",
    "بين": "بَيْنَ",
    "فوق": "فَوْقَ",
    "تحت": "تَحْتَ",
    "امام": "أَمَامَ",
    "خلف": "خَلْفَ",
    "قبل": "قَبْلَ",
    "بعد": "بَعْدَ",
    "عند": "عِنْدَ",
    "اذا": "إِذَا",
    "لو": "لَوْ",
    "لكن": "لَكِنَّ",
    "او": "أَوْ",
    "ثم": "ثُمَّ",
    "حتى": "حَتَّى",
    # Family
    "اب": "أَبٌ",
    "ام": "أُمٌّ",
    "اخ": "أَخٌ",
    "اخت": "أُخْتٌ",
    "ابن": "ابْنٌ",
    "بنت": "بِنْتٌ",
    "زوج": "زَوْجٌ",
    "زوجة": "زَوْجَةٌ",
    "جد": "جَدٌّ",
    "جدة": "جَدَّةٌ",
    # Numbers
    "واحد": "وَاحِدٌ",
    "اثنان": "اثْنَانِ",
    "ثلاثة": "ثَلَاثَةٌ",
    "اربعة": "أَرْبَعَةٌ",
    "خمسة": "خَمْسَةٌ",
    "عشرة": "عَشَرَةٌ",
    # Common verbs (past tense, 3rd person masculine)
    "اكل": "أَكَلَ",
    "شرب": "شَرِبَ",
    "نام": "نَامَ",
    "درس": "دَرَسَ",
    "كتب": "كَتَبَ",
    "قرأ": "قَرَأَ",
    "رأى": "رَأَى",
    "سمع": "سَمِعَ",
    "فهم": "فَهِمَ",
    "عرف": "عَرَفَ",
    "اراد": "أَرَادَ",
    "استطاع": "اسْتَطَاعَ",
    "وجد": "وَجَدَ",
    "بدأ": "بَدَأَ",
    "انتهى": "انْتَهَى",
    "فتح": "فَتَحَ",
    "اغلق": "أَغْلَقَ",
    "خرج": "خَرَجَ",
    "دخل": "دَخَلَ",
    "وصل": "وَصَلَ",
    "رجع": "رَجَعَ",
    "بنى": "بَنَى",
    "صنع": "صَنَعَ",
    "طور": "طَوَّرَ",
    # Common adjectives
    "سريع": "سَرِيعٌ",
    "بطيء": "بَطِيءٌ",
    "قوي": "قَوِيٌّ",
    "ضعيف": "ضَعِيفٌ",
    "غني": "غَنِيٌّ",
    "فقير": "فَقِيرٌ",
    "سهل": "سَهْلٌ",
    "صعب": "صَعْبٌ",
    "مهم": "مُهِمٌّ",
    "خطير": "خَطِيرٌ",
    "غالي": "غَالٍ",
    "رخيص": "رَخِيصٌ",
    "نظيف": "نَظِيفٌ",
    "متعب": "مُتْعِبٌ",
    "مفيد": "مُفِيدٌ",
    "دقيق": "دَقِيقٌ",
    # Technology (common in Mazen's domain)
    "حاسوب": "حَاسُوبٌ",
    "هاتف": "هَاتِفٌ",
    "برنامج": "بَرْنَامَجٌ",
    "تطبيق": "تَطْبِيقٌ",
    "بيانات": "بَيَانَاتٌ",
    "ذكاء": "ذَكَاءٌ",
    "اصطناعي": "اصْطِنَاعِيٌّ",
    "شبكة": "شَبَكَةٌ",
    "نظام": "نِظَامٌ",
    "خوارزمية": "خَوَارِزْمِيَّةٌ",
    "مبرمج": "مُبَرْمِجٌ",
    "مطور": "مُطَوِّرٌ",
    # Places / nature
    "مدينة": "مَدِينَةٌ",
    "بلد": "بَلَدٌ",
    "وطن": "وَطَنٌ",
    "عالم": "عَالَمٌ",
    "ارض": "أَرْضٌ",
    "سماء": "سَمَاءٌ",
    "بحر": "بَحْرٌ",
    "جبل": "جَبَلٌ",
    "صحراء": "صَحْرَاءُ",
    "شارع": "شَارِعٌ",
    "مكتب": "مَكْتَبٌ",
    "مصنع": "مَصْنَعٌ",
    # Food
    "خبز": "خُبْزٌ",
    "لحم": "لَحْمٌ",
    "فاكهة": "فَاكِهَةٌ",
    "قهوة": "قَهْوَةٌ",
    "شاي": "شَايٌ",
    "تمر": "تَمْرٌ",
    "ارز": "أَرُزٌّ",
    # Abstract / common nouns
    "فكرة": "فِكْرَةٌ",
    "مشروع": "مَشْرُوعٌ",
    "هدف": "هَدَفٌ",
    "حل": "حَلٌّ",
    "سبب": "سَبَبٌ",
    "نتيجة": "نَتِيجَةٌ",
    "فرصة": "فُرْصَةٌ",
    "تجربة": "تَجْرِبَةٌ",
    "وقت": "وَقْتٌ",
    "مكان": "مَكَانٌ",
    "طريقة": "طَرِيقَةٌ",
    "معلومة": "مَعْلُومَةٌ",
}

# Explicit forms avoid unsafe orthographic generation (for example بطيئة).
#
# Self-authored, rule-based, deliberately narrow. Standard Arabic feminine
# adjectives/nisba-style nouns are formed by suffixing taa marbuta (ة) to
# the masculine sound form: kabiir -> kabiira, sariia -> sariia+ة, etc.
# That rule is NOT safe for every masculine adjective, so this only ever
# runs against a hand-picked whitelist of words already in _DICTIONARY
# that are known to follow the regular sound pattern - never applied
# blindly to the whole dictionary. Explicitly excluded (would be silently
# wrong if generated): the comparative/color-defect "afAal" pattern
# (احمر -> حمراء, اكبر -> كبرى - irregular، not +ة) and manqus nouns ending
# in a bare ya/kasratan (غالٍ) whose feminine does not follow this suffix
# rule either. Both classes are simply left out of the whitelist below.
_DICTIONARY.update(
    {
        "جميلة": "جَمِيلَة",
        "قبيحة": "قَبِيحَة",
        "كبيرة": "كَبِيرَة",
        "صغيرة": "صَغِيرَة",
        "طويلة": "طَوِيلَة",
        "قصيرة": "قَصِيرَة",
        "جديدة": "جَدِيدَة",
        "قديمة": "قَدِيمَة",
        "سريعة": "سَرِيعَة",
        "بطيئة": "بَطِيئَة",
        "قوية": "قَوِيَّة",
        "ضعيفة": "ضَعِيفَة",
        "غنية": "غَنِيَّة",
        "فقيرة": "فَقِيرَة",
        "سهلة": "سَهْلَة",
        "صعبة": "صَعْبَة",
        "مهمة": "مُهِمَّة",
        "خطيرة": "خَطِيرَة",
        "رخيصة": "رَخِيصَة",
        "نظيفة": "نَظِيفَة",
        "متعبة": "مُتْعِبَة",
        "مفيدة": "مُفِيدَة",
        "دقيقة": "دَقِيقَة",
        "طالبة": "طَالِبَة",
        "معلمة": "مُعَلِّمَة",
        "مهندسة": "مُهَنْدِسَة",
        "طبيبة": "طَبِيبَة",
        "مبرمجة": "مُبَرْمِجَة",
        "مطورة": "مُطَوِّرَة",
    }
)

# Citation forms deliberately omit contextual case endings/tanween.
for _key, _value in list(_DICTIONARY.items()):
    _DICTIONARY[_key] = _value.rstrip("ًٌٍَُِ")

_TOKEN_RE = re.compile(r"([؀-ۿ]+|[^؀-ۿ]+)")


_MARK_RE = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
_SUN_LETTERS = set("تثدذرزسشصضطظلن")


def _lookup(word: str) -> tuple[str | None, bool]:
    """Look up a word, stripping the definite article 'ال' as a fallback.

    Returns (diacritized_form_or_None, matched_via_al_stripping). Prepending
    'الْ' to a dictionary root ignores sun/moon-letter assimilation (real
    Arabic would fuse it into the first consonant for sun letters) - a
    known, disclosed simplification, not a silent inaccuracy: word_report
    marks these as 'matched_via': 'al_stripped' so it's inspectable.
    """
    bare = _MARK_RE.sub("", unicodedata.normalize("NFC", word))
    direct = _DICTIONARY.get(bare)
    if direct:
        return direct, False
    if len(bare) > 4 and bare.startswith("ال"):
        root = bare[2:]
        root_diacritized = _DICTIONARY.get(root)
        if root_diacritized:
            if root[0] in _SUN_LETTERS:
                return "ال" + root_diacritized[0] + "ّ" + root_diacritized[1:], True
            return "الْ" + root_diacritized, True
    return None, False


def diacritize(text: str) -> DiacritizeResult:
    """Restore diacritics for dictionary-known words; leave the rest as-is.

    Returns the diacritized text plus a per-word breakdown and a coverage
    ratio (share of Arabic word-tokens that were actually in the
    dictionary), so callers can tell a well-covered sentence from a mostly
    untouched one instead of assuming full diacritization happened.
    """
    text = validate_text(text)

    pieces = _TOKEN_RE.findall(text)
    out = []
    word_report = []
    arabic_word_count = 0
    covered_count = 0

    for piece in pieces:
        if re.match(r"^[؀-ۿ]+$", piece):
            arabic_word_count += 1
            diacritized, via_al = _lookup(piece)
            if diacritized:
                covered_count += 1
                out.append(diacritized)
                word_report.append(
                    WordReport(
                        word=piece,
                        diacritized=diacritized,
                        in_dictionary=True,
                        matched_via="al_stripped" if via_al else "direct",
                    )
                )
            else:
                out.append(piece)
                word_report.append(WordReport(word=piece, diacritized=piece, in_dictionary=False))
        else:
            out.append(piece)

    coverage = round(covered_count / arabic_word_count, 3) if arabic_word_count else 0.0

    return DiacritizeResult(
        original=text,
        diacritized="".join(out),
        coverage=coverage,
        words_covered=covered_count,
        words_total=arabic_word_count,
        word_report=word_report,
        warnings=["Lexical lookup only; grammatical case endings are not inferred."],
    )
