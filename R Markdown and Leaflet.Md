---
title: "R Markdown and Leaflet"
author: "Hayford Tetteh"
date: "09/06/2020"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Introduction
This is an Coursera project defined as follow= “Create a web page using R Markdown that features a map created with Leaflet.

Host your webpage on either GitHub Pages, RPubs, or NeoCities.

1. Your webpage must contain the date that you created the document, and it must contain a map created with 
2. Leaflet. We would love to see you show off your creativity!"


## Importing required Packages
```{r}
library(leaflet)
library(htmltools)
```


```{r cars}
setwd("C:/Users/TOSHIBA-PC/Desktop/R_Programming/Reproducible/Course 9")
Restaurants <-read.csv("Cape Town Restaurants7.csv",sep=",")
#head(Restaurants)
```

## Accessing the data from local directory
These was obtained using Foursquare API and saved into csv file.

```{r Cape Town Restaurants, echo=FALSE}
my_map <- data.frame(Name = Restaurants$Name,
                     Dis = Restaurants$Distance,
                     Cat = Restaurants$Categories,
                     Lat = Restaurants$Latitude,
                     Long = Restaurants$Longitude,
                     Neigh = Restaurants$Neighborhood,
                     Likes = Restaurants$Likes)
```

```{r}
map <- my_map %>%
  leaflet() %>%
  addTiles() %>%
  addMarkers(popup= paste
             ("<br>Name: ", 
               htmlEscape(Restaurants$Name),
              "<br>Distance: ", 
              htmlEscape(Restaurants$Distance),
              "<br>Categories: ", 
               htmlEscape(Restaurants$Categories), 
              "<br>Neighborhood: ", 
               htmlEscape(Restaurants$Neighborhood),
              "<br>Likes: ",
               formatC(Restaurants$Likes,
               format = "d", big.mark = ",")
        ))
  
    
map
```

