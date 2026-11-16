# Category profiles

The 35 core category profiles (architecture section 3.1) live in `src/shopping_agent/semantics/profiles.yaml`. They are loaded and checked by `shopping_agent.semantics.profiles`:

```python
from shopping_agent.semantics.profiles import load_profiles

milk = load_profiles()["milk"]
```

## A profile

```yaml
milk:
  keywords: [חלב]
  exclude_keywords: [סויה, קוקוס, שוקולד, אבקת, עיזים]
  defaults: {animal: cow, flavor: plain, storage: fresh}
  free: [fat_pct, brand]
  typical_pack: {value: 1000, unit: ml}
tomatoes:
  keywords: [עגבניה, עגבניות]
  exclude_keywords: [רסק, מיובשות]
  defaults: {form: fresh}
  free: [variety]
  typical_pack: {value: 1000, unit: g}
  weighed: true
  typical_unit_weight: {value: 120, unit: g}
```

| Field | Meaning |
|---|---|
| `keywords` | A name belongs to the category when it starts with one of these. A keyword can be two words (`שמן זית`). |
| `exclude_keywords` | ...and none of these words appears in it. |
| `defaults` | Attribute values assumed when the user did not state them. They are hard filters. |
| `free` | Attributes where any value is accepted, so the cheapest wins. |
| `typical_pack` | The usual pack, in `ml`, `g` or `unit`. |
| `weighed` | `true` for a category sold by the kilogram. |
| `typical_unit_weight` | For a weighed category: the weight of one piece, in `g`. |

## The schema check

The file does not load unless:

- every profile writes out `keywords`, `exclude_keywords`, `defaults`, `free` and `typical_pack`;
- `keywords` is not empty, and no list repeats a value;
- a keyword is not also an exclude keyword, and an attribute is not both a default and free;
- attribute and profile names are lower-case `snake_case`;
- a weighed category has a `typical_unit_weight`, and states it and its pack in grams; other categories have none;
- no keyword belongs to two profiles, and no key is written twice.

The check runs with the test suite (`tests/semantics/test_profiles.py`), which also checks that the file holds 35 profiles.

## The 35 profiles

| Group | Profiles |
|---|---|
| Dairy and eggs | milk, cottage, yogurt, white_cheese, yellow_cheese, cream_cheese, butter, eggs |
| Bakery | bread, pita, challah |
| Pantry | rice, pasta, flour, sugar, salt, oil, olive_oil, tuna, tomato_paste, tahini, instant_coffee, turkish_coffee, tea |
| Drinks | water |
| Fruit and vegetables (weighed) | tomatoes, cucumbers, onions, potatoes, carrots, peppers, bananas, apples |
| Meat (weighed) | chicken_breast |
| Household | toilet_paper |

## Changing the file

Add or edit a profile in `profiles.yaml` and run `uv run pytest tests/semantics`. When the number of profiles changes, update the count in the test.
