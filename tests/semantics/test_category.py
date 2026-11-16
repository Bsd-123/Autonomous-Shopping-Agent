"""Tests for category assignment.

Every product name below is a real name from the price files of the three
chains (Yesh Hesed, Osher Ad, Maayan 2000), as published.
"""

import pytest

from shopping_agent.semantics.category import Keywords, assign_category, words

PROFILES = {
    "milk": Keywords(
        ("חלב",), ("סויה", "קוקוס", "שוקולד", "אבקת", "עיזים", "עזים", "וניל", "שקדים")
    ),
    "tomatoes": Keywords(
        ("עגבניה", "עגבניות"), ("רסק", "מיובשות", "מרוסקות", "חתוכות", "קצוצות", "שרי")
    ),
    "oil": Keywords(("שמן",), ("למאור", "טיפולי")),
    "olive_oil": Keywords(("שמן זית",), ("למאור",)),
    "cottage": Keywords(("קוטג'", "קוטג"), ("זיתים", "שום", "שמיר")),
    "potatoes": Keywords(("תפוח אדמה", 'תפו"א'), ("למיקרו",)),
    "apples": Keywords(("תפוח עץ", "תפוחים"), ("יבש", "מיובשים")),
    "salt": Keywords(("מלח",), ("לימון",)),
    "onions": Keywords(("בצל",), ("ירוק", "מטוגן")),
}


def category(name: str | None, brand: str | None = None) -> str | None:
    return assign_category(name, PROFILES, brand=brand)


class TestRealProductNames:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            # Yesh Hesed
            ("חלב תנובה 3% שומן 2 ליטר", "milk"),
            ("חלב 3% מהדרין 2ל תנובה", "milk"),
            ("חלב טרי מהגולן 1% 1 ליטר", "milk"),
            ("קוטג תנובה 3% שומן 250", "cottage"),
            ("שמן קנולה+ויטמיןE ל1", "oil"),
            ("שמן זית כתית מעולה 500 מ", "olive_oil"),
            ('תפו"א לבן ארוז', "potatoes"),
            ("תפוח אדמה אדום תפזורת", "potatoes"),
            ("תפוח עץ סמיט *מולדבה", "apples"),
            ('מלח הארץ גס צנצנת 1 ק"ג', "salt"),
            ("בצל יבש", "onions"),
            ("עגבניה", "tomatoes"),
            # Osher Ad
            ("חלב 3% תנובה 2 ליטר", "milk"),
            ("חלב שקית 3% תנובה 1", "milk"),
            ("קוטג 5% תנובה 250 גר", "cottage"),
            ("שמן קנולה טעמן 1 ליט", "oil"),
            ("תפוח אדמה אדום לאפיה", "potatoes"),
            ("תפוח עץ גאלה ברשת", "apples"),
            # Maayan 2000
            ("חלב תנובה עיזים 3.6%", None),
            ("קוטג' 5% תנובה 250ג' מהדרין", "cottage"),
            ('שמן זית כתית מעולה זיתא 750 מ"ל', "olive_oil"),
            ("עגבניה חממה", "tomatoes"),
            ("תפוחים ברשת", "apples"),
        ],
    )
    def test_products_get_their_category(self, name, expected):
        assert category(name) == expected

    @pytest.mark.parametrize(
        "name",
        [
            # Milk chocolate and other names where milk is not the first word.
            "YOLO שוקולד חלב מעולה",
            "WOW כדור חלב 50 גרם",
            "אינסטנ.פודינג ריבת חלב84",
            "אבקת חלבון חלב + ויט",
            "אוסובוקו עגל חלב טרי",
            # Soy, coconut and goat milk: milk comes first, an exclude keyword follows.
            'חלב קוקוס 250 מ"ל',
            "חלב קוקוס ויליפוד 400ג'",
            "חלב עיזים4%שומן 1ל תנובה",
            "חלב עזים קרטון1ל צוריאל",
            "חלב וניל2% 1ליטר תנובה",
            # A longer word that only starts like a keyword.
            "חלבה מסולסלת שוקו 250 גר' בארכה",
            'WIN חלבון בטעם שוקולד קפה, 330 מ"ל',
        ],
    )
    def test_compound_milk_names_are_not_milk(self, name):
        assert category(name) is None

    def test_soy_milk_is_not_milk_with_or_without_its_brand(self):
        name = "אלפרו חלב סויה וניל 1ל'"

        assert category(name) is None
        assert category(name, brand="אלפרו") is None

    @pytest.mark.parametrize(
        "name",
        [
            "רסק עגבניות 22 בריקס 250",
            "רסק עגבניות גפן 425",
            "מחית עגבניות טבעית3*400ג",
            "נקטר עגבניות 1 ליטר",
            "מארז עגבניות מרוסקות",
            "במיה ברוטב עגבניות ויטה 600ג'",
            "עגבניות מרוסקות800ג יכין",
            "עגבניות שרי אשכולות",
            "עגבניות קצוצות 240 גרם",
            "עגבניות חתוכות בבקבוק",
        ],
    )
    def test_tomato_paste_and_processed_tomatoes_are_not_tomatoes(self, name):
        assert category(name) is None

    @pytest.mark.parametrize(
        "name",
        [
            "שמן זית למאור 1 ליטר",
            "שמן טיפולי קמיל בלו",
            "מלח לימון 700 גרם",
            "בצל ירוק מהדרין",
            "בצל מטוגן פריך 400 גרם",
            "תפוח עץ יבש",
            "תפוחים מיובשים 100 גרם",
            "תפוח אדמה למיקרו )ק(",
            "קוטג עם שום שמיר5% 250ג",
            "קוטג 5% עם זיתים 250 גר",
        ],
    )
    def test_an_exclude_keyword_anywhere_in_the_name_blocks_the_category(self, name):
        assert category(name) is None

    def test_products_outside_every_profile_have_no_category(self):
        assert category("במבה אסם 80 גרם") is None
        assert category("10  צלחות קריסטל קטנות") is None


class TestBrandRemoval:
    def test_a_leading_brand_is_removed_before_the_first_word_is_read(self):
        assert category("יכין מחית עגבניות טבעית  400*3 גר", brand="יכין") is None
        assert category("תנובה חלב 3% 1 ליטר") is None
        assert category("תנובה חלב 3% 1 ליטר", brand="תנובה") == "milk"

    def test_a_brand_of_two_words_is_removed(self):
        assert category("מי עדן חלב 1 ליטר", brand="מי עדן") == "milk"

    def test_a_brand_later_in_the_name_changes_nothing(self):
        assert category("חלב 3% מהדרין 2ל תנובה", brand="תנובה") == "milk"

    def test_a_missing_brand_is_ignored(self):
        assert category("חלב 3% 1 ליטר", brand=None) == "milk"
        assert category("חלב 3% 1 ליטר", brand="  ") == "milk"


class TestListItems:
    """The same function reads what the user wrote."""

    @pytest.mark.parametrize(
        ("item", "expected"),
        [
            ("חלב", "milk"),
            ("חלב 3%", "milk"),
            ("עגבניות", "tomatoes"),
            ("שמן זית", "olive_oil"),
            ("שמן", "oil"),
            ("תפוח אדמה", "potatoes"),
            ("קוטג'", "cottage"),
            ("חלב סויה", None),
            ("שוקולד חלב", None),
            ("רסק עגבניות", None),
            ("במבה", None),
        ],
    )
    def test_items_get_the_category_of_the_products_they_ask_for(self, item, expected):
        assert category(item) == expected

    def test_an_item_and_its_product_land_in_the_same_category(self):
        assert category("חלב 3%") == category("חלב תנובה 3% שומן 2 ליטר") == "milk"
        assert category("חלב סויה") == category("אלפרו חלב סויה וניל 1ל'", brand="אלפרו")


class TestRules:
    def test_the_keyword_must_be_the_first_content_word(self):
        assert category("3% חלב 1 ליטר") == "milk"
        assert category("12 בצל שאלוט 350ג יח") == "onions"
        assert category("טרי חלב") is None

    def test_a_number_glued_to_a_word_does_not_hide_it(self):
        assert words("חלב וניל2% 1ליטר") == ("חלב", "וניל", "2%", "1ליטר")
        assert words("חלב עיזים4%שומן") == ("חלב", "עיזים", "4%שומן")
        assert category("חלב3% 1 ליטר") == "milk"

    def test_the_longer_keyword_wins(self):
        assert category('שמן זית 750 מ"ל') == "olive_oil"
        assert category("תפוח אדמה לבן") == "potatoes"
        assert category("תפוח עץ גאלה") == "apples"

    def test_a_word_is_matched_whole(self):
        # "מלחיה" contains "מלח", and "בשמן" ends with "שמן".
        assert category("מלחיה מעץ") is None
        assert category("בשמן קנולה") is None
        assert category("מלח לימוני 100 גרם") == "salt"

    def test_hebrew_quote_marks_stay_inside_the_word(self):
        assert words("קוטג' 5%") == ("קוטג'", "5%")
        assert words('תפו"א לבן') == ('תפו"א', "לבן")

    def test_latin_text_is_matched_without_regard_to_case(self):
        profiles = {"cola": Keywords(("Coca Cola",), ("zero",))}

        assert assign_category("COCA COLA 1.5L", profiles) == "cola"
        assert assign_category("Coca Cola ZERO 1.5L", profiles) is None

    @pytest.mark.parametrize("name", [None, "", "   ", "500", "3% 1.5", "+ / ,"])
    def test_a_name_without_a_content_word_has_no_category(self, name):
        assert category(name) is None

    def test_no_profiles_means_no_category(self):
        assert assign_category("חלב 3%", {}) is None

    def test_any_object_with_keywords_and_exclude_keywords_works(self):
        class Profile:
            keywords = ["חלב"]
            exclude_keywords = ["סויה"]

        assert assign_category("חלב 3%", {"milk": Profile()}) == "milk"
        assert assign_category("חלב סויה", {"milk": Profile()}) is None
