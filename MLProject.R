# Praise Ogbebor R11835618




# libraries 
library(caret)
library(neuralnet)
library(e1071)
library(adabag)
library(nnet)
library(rpart.plot)
library(ipred)
library(patchwork)
library(ggplot2)



############### Data Inspection #################################
df <- read.csv("financial risk accessment.csv")
df

summary(df)
colnames(df)
str(df)




# Detailed missing data report

## Detect missing values (using is.na() method)
for (col in names(df)) {
  n_MV <- sum(is.na(df[[col]]))
  cat(paste0(col, ': ', n_MV, '\n'))
}



##################### Data Visualization ###########################

# Calculate Mean Income by Gender
gender_income <- aggregate(Income ~ Gender, data = df, FUN = mean)

# Histogram Plot
barplot(
  height = gender_income$Income,
  names.arg = gender_income$Gender,
  col = c("lightblue", "pink", "orange"),
  main = "Average Income by Gender",
  xlab = "Gender",
  ylab = "Average Income"
)



# Plot 1: Risk Rating by Gender
plot1 <- ggplot(df, aes(x = `Risk.Rating`, fill = Gender)) +
  geom_bar(position = "dodge") +
  labs(title = "Risk Rating by Gender", x = "Risk Rating", y = "Count") +
  theme_minimal()

# Plot 2: Risk Rating by Marital Status
plot2 <- ggplot(df, aes(x = `Risk.Rating`, fill = `Marital.Status`)) +
  geom_bar(position = "dodge") +
  labs(title = "Risk Rating by Marital Status", x = "Risk Rating", y = "Count") +
  theme_minimal()

# Plot 3: Risk Rating by Educational level
plot3 <- ggplot(df, aes(x = `Risk.Rating`, fill = `Education.Level`)) +
  geom_bar(position = "dodge") +
  labs(title = "Risk Rating by Educational Level", x = "Risk Rating", y = "Count") +
  theme_minimal()

# Combine the plots side by side
plot1 + plot2 + plot3


# Boxplot: Risk Rating by Gender
plot1 <- ggplot(df, aes(x = `Risk.Rating`, y = Income, fill = Gender)) +
  geom_boxplot(outlier.colour = "red", outlier.shape = 1) +
  labs(title = "Income by Risk Rating and Gender", x = "Risk Rating", y = "Income") +
  theme_minimal()

# Boxplot: Risk Rating by Marital Status
plot2 <- ggplot(df, aes(x = `Risk.Rating`, y = Income, fill = `Marital.Status`)) +
  geom_boxplot(outlier.colour = "red", outlier.shape = 1) +
  labs(title = "Income by Risk Rating and Marital Status", x = "Risk Rating", y = "Income") +
  theme_minimal()

# Boxplot: Risk Rating by Education level
plot3 <- ggplot(df, aes(x = `Risk.Rating`, y = Income, fill = Education.Level)) +
  geom_boxplot(outlier.colour = "red", outlier.shape = 1) +
  labs(title = "Income by Risk Rating and Education Level", x = "Risk Rating", y = "Income") +
  theme_minimal()

# Boxplot: Risk Rating by Marital Status
plot4 <- ggplot(df, aes(x = `Risk.Rating`, y = Income, fill = `Employment.Status`)) +
  geom_boxplot(outlier.colour = "red", outlier.shape = 1) +
  labs(title = "Income by Risk Rating and Marital Status", x = "Risk Rating", y = "Income") +
  theme_minimal()

# Combine the plots side by side

plot1 + plot2 + plot3 + plot4

#################################################
# Goal : To predict the credit risk ratings of an customer
# A classification machine learning algorithm is implemented 
######################################################################

######  check for missing values

###### Data Preprocessing #######

## Risk.Rating column is used my target column column

### create a dummy variable for Risk Rating 
# This is multilevel classification since the category is more than two
df$Risk <- ifelse(df$Risk.Rating == "High", 3, 
                      ifelse(df$Risk.Rating == "Medium", 2, 1))
df


### Drop columns city, state, country, Risk Rating(hard coded and replaced a new column called Risk)
df <- df[, -c(15, 16, 17, 20)]

mydata <- df
mydata

###### Data Cleaning #######

### Column Mapping
# This is important to convert columns with characters entries into numericals 

# Gender mapping
# Define the mapping as a named vector
Gender_map <- c("Male" = 1, "Female" = 2, "Non-binary" = 3)

# Check if the "Gender" column exists and map the values
if ("Gender" %in% colnames(mydata)) {
  mydata$Gender <- as.integer(Gender_map[mydata$Gender])}

# Educational.Level mapping
# Define the mapping as a named vector
Educational_map <- c("PhD" =1, "Master's" = 2, "Bachelor's" = 3,"High School" = 4)

if ("Education.Level" %in% colnames(mydata)) {
  mydata$Education.Level <- as.integer(Educational_map[mydata$Education.Level])}

#  Marital.Status mapping
Marital.Status_map <- c("Divorced" =3,  "Widowed" = 4, "Single" = 1, "Married" = 2)

# Check if the "Marital.Status" column exists and map the values
if ("Marital.Status" %in% colnames(mydata)) {
  mydata$Marital.Status <- as.integer(Marital.Status_map[mydata$Marital.Status])}

# Loan.Purpose mapping

Loan.Purpose_map <- c("Business" = 1, "Auto" = 2, "Home" = 3, "Personal" = 4)

# Check if the "Loan.Purpose" column exists and map the values
if ("Loan.Purpose" %in% colnames(mydata)) {
  mydata$Loan.Purpose <- as.integer(Loan.Purpose_map[mydata$Loan.Purpose])}


# Employment Status mapping
# Define the mapping as a named vector
Employment.Status_map <- c("Employed" =1, "Self-employed" = 2, "Unemployed" = 3)

if ("Employment.Status" %in% colnames(mydata)) {
  mydata$Employment.Status <- as.integer(Employment.Status_map[mydata$Employment.Status])}


#  Payment.History  mapping
Payment.History_map <- c("Excellent" =1,  "Good" = 2, "Fair" = 3, "Married" = 4)

# Check if the "Marital.Status" column exists and map the values
if ("Payment.History" %in% colnames(mydata)) {
  mydata$Payment.History <- as.integer(Payment.History_map[mydata$Payment.History])}

## Verify the mapping
head(mydata)

##### Diagnoses of missing values
## Detect missing values 
for (col in names(mydata)) {
  n_MV <- sum(is.na(mydata[[col]]))
  cat(paste0(col, ': ', n_MV, '\n'))
}


### Diagnose the missing values in Previous.Defaults
#### Diagnosis based on Debt.to.Income.Ratio attribute 

## Compare box plots for Debt.to.Income.Ratio data with vs. without missing values of Previous.Defaults
mydata$Previous.Defaults <- is.na(mydata$Previous.Defaults)

plot1 <- ggplot(mydata, aes(x = as.factor(Previous.Defaults), y = Debt.to.Income.Ratio)) + 
  geom_boxplot() + 
  labs(title = "Debt.to.Income.Ratio vs Missing Previous.Defaults") + 
  xlab("Missing Previous.Defaults")




## Compare box plots for Payment History vs. without missing values of Income
mydata$Income <- is.na(mydata$Income)

plot2 <- ggplot(mydata, aes(x = as.factor(Income), y = Payment.History)) + 
  geom_boxplot() + 
  labs(title = "Payment History vs Missing Income") + 
  xlab("Missing Income")



plot1 + plot2



## t-test to check if there is significant difference
t.test(Debt.to.Income.Ratio ~ Previous.Defaults, data = mydata)
###############################################
#INSIGHT FROM PLOT:

# Given the similarity of the box plots, there is no significant relationship between missing values in Previous.Defaults and the Debt.to.Income.Ratio.
# This suggests that the missingness in Previous.Defaults is unlikely to be systematically related to Debt.to.Income.Ratio 
# and could be considered missing completely at random 
##############################################

#### Diagnose the missing values in Assets.Value  based on some selected attributes
numerical_attributes <- c('Age', 'Debt.to.Income.Ratio', 'Marital.Status.Change', 'Education.Level')
mydata$Assets.Value <- is.na(mydata$Assets.Value)

for (attr in numerical_attributes) {
  cat(paste(attr, "\nt-test p-value: "))
  print(t.test(mydata[[attr]] ~ mydata$Assets.Value)$p.value)
}

## t-test for Employent Status and missing values in Assets.Value
t.test(Employment.Status ~ Assets.Value, data = mydata)
################################################################################
#INSIGHT FROM THE T-TEST:

# There is significant relationship between Missing values in Assets.Value and 
# Age, Debt.to.Income.Ratio, or Education.Level (p-values = 0.8641196, 0.5263258, 0.4789034 > 0.05)


# Interestingly, there is relationship between the Missing values in Assets.Value and 
  
# Marital.Status.Change: Mean differences were statistically significant (p-value = 0.02128165 < 0.05).
# Intuitively,  customers who has had a change in their status may be skeptical in revealing their asset due to the fear of unforseen contigencies.
# Employment.Status: A significant difference in mean values between missing and non-missing groups was observed (p-value = 0.02715 < 0.05).
# Intuitively the customers may be unwilling to disclose their assets probably due for some reasons such profiling etc
################################################################################


####### Handling Missing Values #######
# Random Forest Approach
# This approach of handling was selected due to the size of data having drop city, state, and country which has cardinallity more than 53)

# Convert logical to numeric/factor if needed
# Check current data type of Income column
# Convert all logical columns to numeric or factor
logical_cols <- sapply(mydata, is.logical)

# Convert logical columns to numeric
mydata[logical_cols] <- lapply(mydata[logical_cols], as.numeric)

# Alternatively, convert logical columns to factor
# mydata[logical_cols] <- lapply(mydata[logical_cols], as.factor)

str(mydata)  # Check types before conversion


library(missForest)
set.seed(123)
mydata <- missForest(mydata)$ximp
mydata

## Recheck if there are any missing values left
for (col in names(mydata)) {
  n_MV <- sum(is.na(mydata[[col]]))
  cat(paste0(col, ': ', n_MV, '\n'))
}


#### Detecting Outliers ####

# Prepare the data (remove missing values and select numeric columns)
x <- na.omit(mydata[, sapply(mydata, is.numeric)])

# Compute Mahalanobis distances
xbar <- colMeans(x)                # Mean of each variable
S <- cov(x)                        # Covariance matrix
d2 <- mahalanobis(x, xbar, S)      # Squared Mahalanobis distances

# Calculate positions and chi-squared quantiles
position <- (1:length(d2) - 1/2) / length(d2)
quantiles <- qchisq(position, df = ncol(x))  # Chi-squared quantiles

# Plot the Q-Q plot
plot(quantiles, sort(d2),
     xlab = expression(paste(chi[17]^2, " Quantile")),  
     ylab = "Ordered Squared Distances",
     main = "Chi-Square Q-Q Plot with outliers")
abline(0, 1, col = "red")  # Reference line for the chi-square distribution
#text(quantiles, sort(d2), names(sort(d2)), col = "blue")

# Remove outliers
outlier<- match(lab <- c(2009,14628,9612,3742,8943), rownames(mydata))

mydata_cln <- mydata [-outlier, ]

# recheck for any outliers left

# Prepare the data 
y <- na.omit(mydata_cln[, sapply(mydata_cln, is.numeric)])

# Compute Mahalanobis distances
xbar <- colMeans(y)                # Mean of each variable
S1 <- cov(y)                        # Covariance matrix
d21 <- mahalanobis(y, xbar, S1)     # Squared Mahalanobis distances

# Determine the chi-squared critical value for outlier detection
alpha <- 0.01  # Significance level
cutoff <- qchisq(1 - alpha, df = ncol(y))  # Chi-squared critical value

# Remove outliers
y_clean <- y[d21 <= cutoff, ]  # Keep only rows with distances below the cutoff

# Recalculate positions and chi-squared quantiles for the filtered data
d21_clean <- mahalanobis(y_clean, colMeans(y_clean), cov(y_clean))
position_clean <- (1:length(d21_clean) - 1/2) / length(d21_clean)
quantiles_clean <- qchisq(position_clean, df = ncol(y_clean))  # Chi-squared quantiles

# Plot the Q-Q plot without outliers
plot(quantiles_clean, sort(d21_clean),
     xlab = expression(paste(chi[17]^2, " Quantile")),  
     ylab = "Ordered Squared Distances",
     main = "Chi-Square Q-Q Plot without Outliers")
abline(0, 1, col = "red")  # Reference line for the chi-square distribution


# No outliers was observed 
plot(mydata_cln)


###### Data Visualization ########

# To visualize the correlation between each variables

library(ggcorrplot)

# Calculate the correlation matrix
cor_matrix <- cor(mydata_cln, use = "complete.obs")

# Create the correlation plot
ggcorrplot(cor_matrix,
           method = "circle",    
           type = "full",        
           lab = FALSE,          
           colors = c("red", "white", "blue"), 
           outline.col = "black" 
) + 
  ggtitle("Correlation Matrix Heatmap") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1), # Make x-axis labels vertical
    axis.text.y = element_text(vjust = 0.5)                         # Adjust y-axis labels
  )


# Create a boxplot of the cleaned data
boxplot(mydata_cln, main = "Boxplot")




#####################################################################
# Goal : To predict the credit risk ratings of an customer
# A classification machine learning algorithm is implemented 
######################################################################

### Partitioning my data
train_index <- sample(rownames(mydata_cln), dim(mydata_cln)[1] * 0.8)
valid_index <- setdiff(rownames(mydata_cln), train_index)
training_data <- mydata_cln[train_index, ]
validation_data <- mydata_cln[valid_index, ]


##### Model Selections ##########
#### Multinomial Logistic Regression model
# This model was selected due to the fact that a 3-level prediction is required(3,2,1) as opposed to binomial(1 vs 0)

### Multinomial regression model
logit_multi <- multinom(Risk ~ ., data = training_data, family = "multinomial")
summary(logit_multi)  # Regression result

### Prediction using validation data
predicted_logit <- predict(logit_multi, newdata = validation_data, type = "class")

### Confusion matrix
# Convert predictions and actual values to factors
predicted_logit <- as.factor(predicted_logit)
actual_risk <- as.factor(validation_data$Risk)

# Generate confusion matrix
confusionMatrix(predicted_logit, actual_risk)
Conf_lr <- confusionMatrix(predicted_logit, actual_risk) 

###### Random forest
library(randomForest)


training_data$Risk = as.factor(training_data$Risk)
validation_data$Risk = as.factor(validation_data$Risk)

random_forest_model = randomForest(Risk ~ ., data = training_data)

### Predict using the validation data
predicted_rf = predict(random_forest_model, newdata = validation_data)

### Confusion matrix
confusionMatrix(predicted_rf, validation_data$Risk) # I didnt have to use relevel which specifically used for binary classification
Conf_rf <- confusionMatrix(predicted_rf, validation_data$Risk)

###### Tree analysis
Tree_model <- rpart(Risk ~ ., data = training_data, method = "class")
prp(Tree_model) # Tree model result 

### Prediction using validating data 
predicted_tree <- predict(Tree_model, newdata = validation_data, type = "class")

# Confusion matrix
confusionMatrix(predicted_tree, validation_data$Risk)
Conf_ta <- confusionMatrix(predicted_tree, validation_data$Risk) 

##### Support Vector Machine(SMV)
training_data$Risk <- as.factor(training_data$Risk)
validation_data$Risk <- as.factor(validation_data$Risk)

# Training
SVM <- svm(Risk ~., data = training_data)

# Prediction
predicted_svm <- predict(SVM, validation_data)

# Confusion matrix
confusionMatrix(predicted_svm, validation_data$Risk)
Conf_svm <- confusionMatrix(predicted_svm, validation_data$Risk)


### Ensemble 
# Ensure Risk is a factor
training_data$Risk <- as.factor(training_data$Risk)
validation_data$Risk <- as.factor(validation_data$Risk)



## Bagging
bag <- bagging(Risk ~ ., data = training_data)  # Train the bagging model
predicted_bagging <- predict(bag, validation_data, type = "class")  # Make predictions

## Confusion Matrix
confusionMatrix(as.factor(predicted_bagging), validation_data$Risk)  # Evaluate the model
Conf_Bg <- confusionMatrix(as.factor(predicted_bagging), validation_data$Risk)


## Boosting 

library(xgboost)
x_train <- model.matrix(~ . - Risk, data = training_data)
# Create validation data matrix
x_validation <- model.matrix(~ . - Risk, data = validation_data)

y_train <- as.numeric(training_data$Risk) - 1

Boosting <- xgboost(data = x_train, label = y_train, nrounds = 90, objective = "multi:softmax", num_class = 3 )  # 90 was obtain to be the optimal rounds of iteration   

predicted <- predict(Boosting, x_validation)
predicted_Boosting <- factor(predicted, labels = levels(training_data$Risk))
confusionMatrix(predicted_Boosting, validation_data$Risk)
Conf_Bst <- confusionMatrix(predicted_Boosting, validation_data$Risk)



### Stacking 

library(caret)
library(nnet)

# Ensureds Risk is a factor
training_data$Risk <- as.factor(training_data$Risk)
validation_data$Risk <- as.factor(validation_data$Risk)

# Partition the training data into two sets for primary and meta models
set.seed(123)  # For reproducibility
train1.index <- sample(1:nrow(training_data), nrow(training_data) * 0.5)
train2.index <- setdiff(1:nrow(training_data), train1.index)

train1.df <- training_data[train1.index, ]
train2.df <- training_data[train2.index, ]

# Train primary models
prim_tree <- train(Risk ~ ., data = train1.df, method = "rpart")
prim_knn <- train(Risk ~ ., data = train1.df, method = "knn", preProcess = c('center', 'scale'))
prim_logit <- train(Risk ~ ., data = train1.df, method = "multinom")  # Multi-class logistic regression

# Make predictions for train2.df
tree_prediction <- predict(prim_tree, newdata = train2.df)
knn_prediction <- predict(prim_knn, newdata = train2.df)
logit_prediction <- predict(prim_logit, newdata = train2.df)

# Compile meta-data for the meta model
meta_data <- data.frame(
  tree_prediction = as.factor(tree_prediction),
  knn_prediction = as.factor(knn_prediction),
  logit_prediction = as.factor(logit_prediction),
  Risk = train2.df$Risk
)

# Train meta model
meta.model <- train(Risk ~ ., data = meta_data, method = "multinom")

# Validate the stacking model
tree_pred_v <- predict(prim_tree, newdata = validation_data)
knn_pred_v <- predict(prim_knn, newdata = validation_data)
logit_pred_v <- predict(prim_logit, newdata = validation_data)

# Prepare meta-data for validation
meta_data_v <- data.frame(
  tree_prediction = as.factor(tree_pred_v),
  knn_prediction = as.factor(knn_pred_v),
  logit_prediction = as.factor(logit_pred_v)
)

# Predict using the meta model
meta_prediction <- predict(meta.model, newdata = meta_data_v)

# Evaluate performance
confusionMatrix(meta_prediction, validation_data$Risk)
Conf_meta <- confusionMatrix(meta_prediction, validation_data$Risk)



######### Model Evaluations and result comparison
# Print all metrics (Accuracy, Precision, Recall, F1-score)


# Function to extract macro-averaged Precision, Recall, and F1-Score
calculate_macro_metrics <- function(conf_matrix) {
  precision <- mean(conf_matrix$byClass[, "Precision"], na.rm = TRUE) 
  recall <- mean(conf_matrix$byClass[, "Recall"], na.rm = TRUE)       
  f1_score <- mean(conf_matrix$byClass[, "F1"], na.rm = TRUE)         
  
  list(
    Precision = precision,
    Recall = recall,
    F1_Score = f1_score
  )
}

# Combine metrics into a data frame
create_metrics_dataframe <- function() {
  model_names <- c("Logistic Regression", "Random Forest", "Boosting", "Bagging", "Meta Model", "SVM")
  accuracies <- c(
    Conf_lr$overall['Accuracy'],
    Conf_rf$overall['Accuracy'],
    Conf_Bst$overall['Accuracy'],
    Conf_Bg$overall['Accuracy'],
    Conf_meta$overall['Accuracy'],
    Conf_svm$overall['Accuracy']
  )
  
  macro_metrics <- lapply(list(Conf_lr, Conf_rf, Conf_Bst, Conf_Bg, Conf_meta, Conf_svm), calculate_macro_metrics)
  
  data.frame(
    Model = model_names,
    Accuracy = unlist(accuracies),
    Precision = sapply(macro_metrics, function(x) x$Precision),
    Recall = sapply(macro_metrics, function(x) x$Recall),
    F1_Score = sapply(macro_metrics, function(x) x$F1_Score)
  )
}

# Create and display the data frame
evaluation_metrics_df <- create_metrics_dataframe()
print(evaluation_metrics_df)
View(evaluation_metrics_df)
