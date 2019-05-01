# This file details the finds of comparing the name of a restaurant to
# the probability of its closure.
#
# The names are formatted into a simple contains/does not contain matrix
# of one-hot vectors. This are then given to a gradient boosting machine
# with the closure (0/1) as Y.
#
# This is done for both hard and soft closures (license and inspection datasets)
import src.fetch as fetch

restaurant_insp = fetch.fetch_restaurant_inspection_data()
insp = fetch.fetch_inspection_data(True)
