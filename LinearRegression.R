# Loading
data(iris)
# Print the first 6 rows
head(iris,6)
df <- iris


# Load the Caret package which allows us to partition the data
library(caret)
# We use to create a partition (80% training 20% testing)
index <- createDataPartition(df$Species, p=0.80, list=FALSE)
# select 20% of the data for testing
testset <- df[-index,]
# select 80% of data to train the models
trainset <- df[index,]

# Run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

# a) linear algorithms
set.seed(7)
fit.lda <- train(Species~., data=trainset, method="lda", metric=metric, trControl=control)
# b) nonlinear algorithms
# CART
set.seed(7)
fit.cart <- train(Species~., data=trainset, method="rpart", metric=metric, trControl=control)
# kNN
set.seed(7)
fit.knn <- train(Species~., data=trainset, method="knn", metric=metric, trControl=control)
# c) advanced algorithms
# SVM
set.seed(7)
fit.svm <- train(Species~., data=trainset, method="svmRadial", metric=metric, trControl=control)
# Random Forest
set.seed(7)
fit.rf <- train(Species~., data=trainset, method="rf", metric=metric, trControl=control)

# summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)


dotplot(results)


print(fit.lda)

predictions <- predict(fit.lda, testset)
confusionMatrix(predictions, testset$Species)