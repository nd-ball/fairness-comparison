library(readr)

library(corrplot)
library(dplyr)

D <- read_csv("covidcases.csv") %>%
  select(aboveAverageBlack,
         aboveNatlAverage, 
         aboveStateAverage,
         majorityFemale,
         aboveAverageOver65,
         Y,
         percent65over,
         percentWomen,
         percentBlack)


corrplot(cor(D), method="number")
