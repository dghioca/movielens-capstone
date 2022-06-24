#MovieLens Capstone Project - Dana Ghioca Robrecht, June 2022

# Data preparation

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#title = as.character(title),
#genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


##########################################################

#	Data exploration and visualization

# Before splitting the edx dataset into training and testing sets,
# I performed some descriptive statistics to get a better idea of the structure of the datasets.
str(edx)
summary(edx)
dim(edx)
n_distinct(edx$movieId)
n_distinct(edx$userId)

str(validation)
summary(validation)
dim(validation)
n_distinct(validation$movieId)
n_distinct(validation$userId)

#I can see that the ratings vary between 0.5 and 5 with an average of 3.512 and a median of 4 for both datasets.

#I noticed that the timestamp (the date and time a movie was reviewed by a user)
#is not very helpful in the current format,
# so I extracted the date and rounded it to week units.
# I have done this on both the edx and validation sets and checked that the 
# same number of dates are included in both sets.

library(lubridate)
edx <- edx %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week"))

validation <- validation  %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week"))

validation <- validation %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId") 

#I also noticed that the year when a movie was released is included in the title, so
# I extracted the release year and created a new column named "released" in the edx and validation sets.
released <- as.numeric(str_sub(edx$title, start = -5, end = -2))
edx <- edx %>% mutate(released = released)
head(edx)
dim(edx)

released <- as.numeric(str_sub(validation$title, start = -5, end = -2))
validation <- validation %>% mutate(released = released)
head(validation)
dim(validation)


#Data visualization of ratings distribution on the edx set.

#Since the ratings are not really continuous data, 
#but rather categorical data because they are grouped in 10 categories, a bar plot was my choice 
#for visualizing the rating distribution instead of a histogram (appropriate for continuous variables).

qplot(rating, data = edx, color = I("black"), geom = "bar")

#The most frequent rating was 4, followed by 3, and then 5,
# indicating a possible bias towards higher ratings.

#I also looked at the distribution of other variables in the dataset 
# as these are the potential effects to account for in my models.

range(edx$date)
qplot(date, data = edx, bins = 40, color = I("black"))
# The dates of review range from 1995 to 2009 and have an interesting distribution 
# showing that there was a review peak around 1996, then another
# surge around 2000-2001, and then again one in 2005.
#(One may wonder, is there a five-year review resurgence pattern?)

range(edx$released)
qplot(released, data = edx, binwidth = 1, color = I("black"))
# Movies were released between 1915 and 2008, but the distribution is skewed to the left
#with a peak in the mid to late 90's suggesting these were most popular movies.

edx %>% separate_rows(genres, sep = "\\|") %>%
  ggplot(aes(genres)) +
  geom_bar() +
  theme(axis.text.x = element_text(angle=90))
# After separating into individual genres the original genre category which 
#includes multiple designations for each movie, the most commonly rated movie 
# genres were Drama, Comedy, and Action and least rated were IMAX, 
#Documentary, and Film-Noir.


##########################################################
# Building the model & Results

#My approach for building the algorithm starts with the models described in Chapter 8 
#in the "Recommendation systems" chapter.

#I started by partitioning the edx data set in training and a test sets, similar to how we split 
#the movielens set but using 20% of the edx dataset for testing.

set.seed(1, sample.kind="Rounding")
index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
edx_train <- edx[-index,]
edx_temp <- edx[index,]  

# Make sure userId and movieId in test set are also in train set
edx_test <- edx_temp %>% 
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")

# Add rows removed from test set back into train set
removed <- anti_join(edx_temp, edx_test)
edx_train <- rbind(edx_train, removed)

rm(index, edx_temp, removed)


#The simplest model assumes that all movies and users produce the same rating. 
#The observed variability in the ratings in the data set is then due only to random variation. 
# I calculated the LSE for this "true" rating mean by calculating the average of the ratings 
#in the edx training set.

#Calculate naive mean on the edx training set.
mu <- mean(edx_train$rating)
mu

naive_rmse <- RMSE(edx_test$rating, mu)
naive_rmse
#The naive RMSE is 1.059904.

#I created a table to keep track of the model improvements.
rmse_results <- tibble(method ="Basic Average", RMSE = naive_rmse)
rmse_results

#In the next step I added in the model a movie bias. 
#I added the new RMSE to the table.
movie_avgs <- edx_train %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu)) 

predicted_ratings_1 <- mu + edx_test %>%
  left_join(movie_avgs, by = "movieId")  %>%
  .$b_i

model_1_rmse <- RMSE(predicted_ratings_1, edx_test$rating)
rmse_results <- bind_rows(rmse_results, 
                          data_frame(method="Model 1 - Movie Effect Model", 
                                     RMSE = model_1_rmse))
rmse_results %>% knitr::kable()

#Adding a movie bias factor in the model decreased the RMSE to 0.94374. 
#Next, I added in the model a user bias.

user_avgs <- edx_train  %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i)) 

predicted_ratings_2 <- edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

model_2_rmse <- RMSE(predicted_ratings_2, edx_test$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Model 2 - Movie + User Effects Model",  
                                     RMSE = model_2_rmse))
rmse_results %>% knitr::kable()
 
#Adding a user bias in the model further decreased the RMSE to 0.86593.
#For the next model, I added a genre bias.

genre_avgs <- edx_train  %>%    
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId')%>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u))

genre_avgs %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

predicted_ratings_3 <- edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  .$pred

model_3_rmse <- RMSE(predicted_ratings_3, edx_test$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Model 3 - Movie + User + Genre Effects Model",  
                                     RMSE = model_3_rmse))
rmse_results %>% knitr::kable()

# This model with three factors has an RMSE of 0.86559, better than the previous one.
# Next, I added the date of review effect to the previous model.

date_avgs <- edx_train %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  group_by(date) %>%
  summarize(b_d = mean(rating - mu - b_i - b_u - b_g))

predicted_ratings_4 <- edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(date_avgs, by='date') %>%
  mutate(pred = mu + b_i + b_u + b_g + b_d) %>%
  .$pred

model_4_rmse <- RMSE(predicted_ratings_4, edx_test$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Model 4 - Movie + User + Genre + Date Effects Model",  
                                     RMSE = model_4_rmse))
rmse_results %>% knitr::kable()


#Adding the date of review modestly reduced the RMSE of the model to 0.86549. 
#Thus, I added lastly the year of release to the model.

release_avgs <- edx_train %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(date_avgs, by='date') %>%
  group_by(released) %>%
  summarize(b_r = mean(rating - mu - b_i - b_u - b_g - b_d))

predicted_ratings_5 <- edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(date_avgs, by='date') %>%
  left_join(release_avgs, by ="released") %>%
  mutate(pred = mu + b_i + b_u + b_g + b_d + b_r) %>%
  .$pred

model_5_rmse <- RMSE(predicted_ratings_5, edx_test$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Model 5 - Movie + User + Genre + Date + Release Effects Model",  
                                     RMSE = model_5_rmse ))
rmse_results %>% knitr::kable()


#So far this was the best model with an RMSE of 0.86526, but still this was higher  
#than the target of 0.86490. Thus, I performed regularization on this best model. 
#I used cross-validation to find the lambda that minimized the RMSE. 

lambdas <- seq(0, 10, 0.5)

rmses <- sapply(lambdas, function(l){
  
      mu <- mean(edx_train$rating)
      
      b_i_r <- edx_train %>% 
      group_by(movieId) %>%
      summarize(b_i_r = sum(rating - mu)/(n()+l))
     
      b_u_r <- edx_train %>% 
      left_join(b_i_r, by="movieId") %>%
      group_by(userId) %>%
      summarize(b_u_r = sum(rating - b_i_r - mu)/(n()+l))
      
      b_g_r <- edx_train %>% 
        left_join(b_i_r, by="movieId") %>%
        left_join(b_u_r, by="userId") %>%
        group_by(genres) %>%
        summarize(b_g_r = sum(rating - b_i_r - b_u_r - mu)/(n()+l))

      b_d_r <- edx_train %>%
        left_join(b_i_r, by="movieId") %>%
        left_join(b_u_r, by="userId") %>%
        left_join(b_g_r, by="genres") %>%
        group_by(date) %>%
        summarize(b_d_r = sum(rating - b_i_r - b_u_r - b_g_r - mu)/(n()+l))
      
      b_r_r <- edx_train %>%
        left_join(b_i_r, by="movieId") %>%
        left_join(b_u_r, by="userId") %>%
        left_join(b_g_r, by="genres") %>%
        left_join(b_d_r, by='date') %>%
        group_by(released) %>%
        summarize(b_r_r = mean(rating -b_i_r - b_u_r - b_g_r - b_d_r - mu)/(n()+l))
      
      predicted_ratings_6 <- edx_test %>% 
      left_join(b_i_r, by = "movieId") %>%
      left_join(b_u_r, by = "userId") %>%
      left_join(b_g_r, by ="genres") %>%
      left_join(b_d_r, by ="date") %>%
      left_join(b_r_r, by ="released") %>%
      mutate(pred = mu + b_i_r + b_u_r + b_g_r + b_d_r + b_r_r) %>%
      .$pred
        
       return(RMSE(predicted_ratings_6, edx_test$rating))
       })

qplot(lambdas, rmses)  
lambda <- lambdas[which.min(rmses)]
lambda

model_6_rmse <- min(rmses)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Model 6 - Regularized Movie + User + Gender + Date + Release Effects Model",  
                                     RMSE = model_6_rmse))
rmse_results %>% knitr::kable()

#The optimal lambda was 5.0 and the RMSE of the regularized model with this 
#lambda was 0.86479, which was below the target of 0.86490!
  

##########################################################
#Testing the algorithm with validation set

  # For the final check, I used the validation set to calculate the predicted 
#ratings which I then compared to the actual ratings in this validation set to obtain the RMSE. 
  
  l <- lambda
  mu_edx <- mean(edx$rating)
  b_i_r <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i_r = sum(rating - mu_edx)/(n()+l))
  
  b_u_r <- edx %>% 
    left_join(b_i_r, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u_r = sum(rating - b_i_r - mu_edx)/(n()+l))
  
  b_g_r <- edx %>% 
    left_join(b_i_r, by="movieId") %>%
    left_join(b_u_r, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g_r = sum(rating - b_i_r - b_u_r - mu_edx)/(n()+l))
  
  b_d_r <- edx %>%
    left_join(b_i_r, by="movieId") %>%
    left_join(b_u_r, by="userId") %>%
    left_join(b_g_r, by="genres") %>%
    group_by(date) %>%
    summarize(b_d_r = sum(rating - b_i_r - b_u_r - b_g_r - mu_edx)/(n()+l))
  
  b_r_r <- edx %>%
    left_join(b_i_r, by="movieId") %>%
    left_join(b_u_r, by="userId") %>%
    left_join(b_g_r, by="genres") %>%
    left_join(b_d_r, by='date') %>%
    group_by(released) %>%
    summarize(b_r_r = mean(rating -b_i_r - b_u_r - b_g_r - b_d_r - mu_edx)/(n()+l))
  
  predicted_ratings_final <- validation %>% 
    left_join(b_i_r, by = "movieId") %>%
    left_join(b_u_r, by = "userId") %>%
    left_join(b_g_r, by ="genres") %>%
    left_join(b_d_r, by ="date") %>%
    left_join(b_r_r, by ="released") %>%
    mutate(pred = mu_edx + b_i_r + b_u_r + b_g_r + b_d_r + b_r_r) %>%
    .$pred
  
  final_rmse_check <- RMSE(predicted_ratings_final, validation$rating)
  final_rmse_check 
  
  # The rmse of my regularized model using the validation set was 0.86431,
  # which is below the 0.86490 threshold.
  