#
# Cole Smith
# Restaurant Closure Engine
# BDS - Undergraduate
# main.py
#

from src.fetch import fetch_restaurant_inspection_data, fetch_inspection_data, fetch_legally_operating_businesses

df = fetch_restaurant_inspection_data()
# df = fetch_restaurant_violation_lookup_table()
# df = fetch_inspection_data()
df2 = fetch_legally_operating_businesses()
