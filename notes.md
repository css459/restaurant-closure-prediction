# Project Update March 21st

## Completed Tasks

- Economic data (VIX, DJIA) dataset is now complete, with information at the **monthly** level

- Demographic data (race and income distribution) is now complete at the **zip code** level.

- The **Restaurant Inspections** dataset is now clean, each row is a **restaurant** with **is_closed**
column added (details below)

- The general **Inspections** dataset is now clean, and filtered by **restaurants, Sidewalk Cafes, and
Gaming Cafes**. The **is_closed** column is appended.

## Current To-Do

- Clean and generate closure reports for the **Legally Operating Businesses" dataset.

- Create merged **Closure** dataset where each row is a closure, and has columns:

    - zip code
    - year
    - month
    - type (reason for closure)
    - (maybe also business type? ie: is_sidewalk_cafe)

- Explore data scraping time requirements and difficulty assessment for Yelp and Google.

## Roadblocks

- **The restaurant inspections does not contain any closed restaurants by definition** (One may
only find [potentially temporary] closures from health inspection shutdowns)

- **Joining the tables on identifiers might be impossible, there are no common identifiers** (Solutions
to this may include fuzzy joining on DBA and address, or using an aggregate [per month instead of per
restaurant] on closures and comparing results

## Findings

- The **Restaurant Inspections** dataset alone did **not** provide any classification models with
meaningful insights. The dataset is extremely imbalanced, and resulted in a near-zero Cohen Kappa value.

- The **Inspections** and **Legally Operating Businesses** data sets are updated on the same day

    - A business must have a valid license

    - We know a business exists if it passes an inspection

## Remarks

- **Soft Closures**: Refer to closures which occur from being condemned by the health department. These
restaurants may re-open at some point in the future and represent a differnt kind of closure than the
kind originally proposed. This information is still useful however, and it may provide insights into
the exact reasons for closures.

- **Hard Closures**: Refer to closures in which the business is not able to undergo required inspection.
If this is the case, a column in **Inspections** will indicate whether the business is out-of-business,
or could not be located. These I am referring to as a hard, voluntary closure which better fits with
the original definition. The issue is that I cannot map these results to any other tables with more
predictive power.

