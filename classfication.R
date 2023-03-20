
library("data.table")
library("mlr3verse")
library("tidyverse")
library("ggplot2")
library("GGally")
library("keras")
library("tidymodels")

# Load the data
set.seed(810)
bank_personal_loan <- readr::read_csv("https://www.louisaslett.com/Courses/MISCADA/bank_personal_loan.csv")
skimr::skim(bank_personal_loan)

# Clean the data
bank_personal_loan["Personal.Loan"] <- lapply(bank_personal_loan["Personal.Loan"] , factor)
bank_personal_loan <- bank_personal_loan|>
  select(-ZIP.Code)
bank_personal_loan["Experience"][bank_personal_loan["Experience"] < 0] <- NA
bank_personal_loan <- na.omit(bank_personal_loan)

# Convert some numeric feature to factor
bank_personal_loan2 <- bank_personal_loan |>
  mutate(Education = as.factor(Education),
         Family = as.factor(Family))



# Task
bank_personal_loan_task <- TaskClassif$new(id = "A",
                                           backend = bank_personal_loan, 
                                           target = "Personal.Loan",
                                           positive = "1")
set.seed(123)
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(bank_personal_loan_task)
set.seed(123)
bootstrap <- rsmp("bootstrap", repeats = 10)
bootstrap$instantiate(bank_personal_loan_task)

lrn_lda <- lrn("classif.lda", predict_type = "prob")
lrn_log_reg <- lrn("classif.log_reg", predict_type = "prob")
lrn_qda <- lrn("classif.qda", predict_type = "prob")
lrn_cart <- lrn("classif.rpart", predict_type = "prob",cp=0)

set.seed(123)
res1 <- benchmark(data.table(
  task       = list(bank_personal_loan_task),
  learner    = list(lrn_lda,lrn_log_reg,lrn_qda,lrn_cart),
  resampling = list(cv5)
), store_models = TRUE) 

res2 <- benchmark(data.table(
  task       = list(bank_personal_loan_task),
  learner    = list(lrn_lda,lrn_log_reg,lrn_qda,lrn_cart),
  resampling = list(bootstrap)
),store_models = TRUE) 

rbind(res1$aggregate(list(msr("classif.ce"),msr("classif.acc")))[1],res2$aggregate(list(msr("classif.ce"),msr("classif.acc")))[1],
      res1$aggregate(list(msr("classif.ce"),msr("classif.acc")))[2],res2$aggregate(list(msr("classif.ce"),msr("classif.acc")))[2],
      res1$aggregate(list(msr("classif.ce"),msr("classif.acc")))[3],res2$aggregate(list(msr("classif.ce"),msr("classif.acc")))[3],
      res1$aggregate(list(msr("classif.ce"),msr("classif.acc")))[4],res2$aggregate(list(msr("classif.ce"),msr("classif.acc")))[4])


# Also, we want to know wether numeric or factor may have heigher accuracy.

bank_personal_loan_task2 <- TaskClassif$new(id = "B",
                                            backend = bank_personal_loan2, 
                                            target = "Personal.Loan",
                                            positive = "1")
set.seed(123)
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(bank_personal_loan_task)
lrn_cart <- lrn("classif.rpart", predict_type = "prob",cp=0)

set.seed(123)
res3 <- benchmark(data.table(
  task       = list(bank_personal_loan_task2),
  learner    = list(lrn_cart),
  resampling = list(cv5)
), store_models = TRUE) 

res3$aggregate(list(msr("classif.ce"),msr("classif.acc")))[1]


# Test/Trian
set.seed(625) 
# First get the training
bank_split <- initial_split(bank_personal_loan)
bank_train <- training(bank_split)

# Then further split the training into validate and test
bank_split2 <- initial_split(testing(bank_split), 0.5)
bank_validate <- training(bank_split2)
bank_test <- testing(bank_split2)

bank_train_task <- TaskClassif$new(id = "Train",
                                   backend = data.frame(bank_train),
                                   target = "Personal.Loan",
                                   positive = "1")


bank_test_task <- TaskClassif$new(id = "Test",
                                  backend = data.frame(bank_test),
                                  target = "Personal.Loan",
                                  positive = "1")
set.seed(123)
lrn_cart$train(bank_train_task)
pred_cart <- lrn_cart$predict(bank_test_task)
pred_cart$score(msr("classif.acc"))

We took one result from cv[1]:

trees <- res1$resample_result(4)

# Then, let's look at the tree from first CV iteration, for example:
tree1 <- trees$learners[[1]]
tree1_rpart <- tree1$model
plot(tree1_rpart, compress = TRUE, margin = 0.1)
text(tree1_rpart, use.n = TRUE, cex = 0.8)


library(caret)

set.seed(625) 
paramGrid <- expand.grid(.cp = seq(0.01, 0.5, 0.01))
control <- trainControl(method="cv", number=5)
metric <- "Accuracy"
set.seed(123)
model <- train(Personal.Loan ~ ., data=bank_personal_loan, method="rpart", 
               trControl=control, metric=metric, 
               tuneGrid=paramGrid)
print(model)

lrn_cart_cp <- lrn("classif.rpart", predict_type = "prob", cp = 0.01)

set.seed(123)
res4 <- benchmark(data.table(
  task       = list(bank_personal_loan_task),
  learner    = list(lrn_cart_cp),
  resampling = list(cv5)
), store_models = TRUE)

rbind(res1$aggregate(list(msr("classif.ce"),
                          msr("classif.acc"),
                          msr("classif.auc"),
                          msr("classif.fpr"),
                          msr("classif.fnr"))),
      res4$aggregate(list(msr("classif.ce"),
                          msr("classif.acc"),
                          msr("classif.auc"),
                          msr("classif.fpr"),
                          msr("classif.fnr"))))


# Neural Network
cake <- recipe(Personal.Loan ~ ., data = bank_personal_loan) %>%
  # step_impute_mean(all_numeric()) %>% # impute missings on numeric values with the mean
  step_center(all_numeric()) %>% # center by subtracting the mean from all numeric features
  step_scale(all_numeric()) %>% # scale by dividing by the standard deviation on all numeric features
  step_unknown(all_nominal(), -all_outcomes()) %>% # create a new factor level called "unknown" to account for NAs in factors, except for the outcome (response can't be NA)
  step_dummy(all_nominal(), one_hot = TRUE) %>% # turn all factors into a one-hot coding
  prep(training = bank_train) # learn all the parameters of preprocessing on the training data

bank_train_final <- bake(cake, new_data = bank_train) # apply preprocessing to training data
bank_validate_final <- bake(cake, new_data = bank_validate) # apply preprocessing to validation data
bank_test_final <- bake(cake, new_data = bank_test) # apply preprocessing to testing data

# part3
bank_train_x <- bank_train_final %>%
  select(-starts_with("Personal.Loan_")) %>%
  as.matrix()
bank_train_y <- bank_train_final %>%
  select(Personal.Loan_X0) %>%
  as.matrix()

bank_validate_x <- bank_validate_final %>%
  select(-starts_with("Personal.Loan_")) %>%
  as.matrix()
bank_validate_y <- bank_validate_final %>%
  select(Personal.Loan_X0) %>%
  as.matrix()

bank_test_x <- bank_test_final %>%
  select(-starts_with("Personal.Loan_")) %>%
  as.matrix()
bank_test_y <- bank_test_final %>%
  select(Personal.Loan_X0) %>%
  as.matrix()


# Part4
set.seed(625)
deep.net <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = "relu",
              input_shape = c(ncol(bank_train_x))) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")
# Have a look at it
deep.net

set.seed(625)
deep.net %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)


set.seed(625)
deep.net %>% fit(
  bank_train_x, bank_train_y,
  epochs = 50, batch_size = 32,
  validation_data = list(bank_validate_x, bank_validate_y),
)

# To get the probability predictions on the test set:
pred_test_prob <- deep.net %>% predict(bank_test_x)

# To get the raw classes (assuming 0.5 cutoff):
pred_test_res <- deep.net %>% predict(bank_test_x) %>% `>`(0.5) %>% as.integer()

table(pred_test_res, bank_test_y)
yardstick::accuracy_vec(as.factor(bank_test_y),
                        as.factor(pred_test_res))
yardstick::roc_auc_vec(factor(bank_test_y, levels = c("1","0")),
                       c(pred_test_prob))

# ROC
library(mlr3)
library(mlr3viz)

# Load benchmark results
res <- res4 # Load benchmark results here

# Create plot object for false negatives
plot_obj <- autoplot(res, type = "roc", measure = msr("classif.fnr"))

# Add title and axis labels
plot_obj + ggtitle("False Negatives Plot") +
  xlab("False Positive Rate") + ylab("True Positive Rate")


# Precision-Recall
autoplot(res4, type = "prc")+ theme_bw()




