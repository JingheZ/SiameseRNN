# # install.packages('SGL')
# install.packages('pROC')
# install.packages('caret')
# install.packages('glmnet')
library(SGL)
library(pROC)
library(caret)
library(glmnet)
library(randomForest)
# load 
setwd('~/PycharmProjects/SiameseRNN/')
train_ids <- read.csv('./data/train_ids.csv', header = FALSE)
test_ids <- read.csv('./data/test_ids.csv', header = FALSE)

read_data <- function(filename, s) {
  dt <- read.csv(filename)
  if (s == 'bps') {
    colnames(dt)[1] <- 'ptid'
  }
  train <- dt[dt$ptid %in% train_ids$V1, ]
  train <- train[sample(nrow(train)),]
  # train$response <- factor(train$response)
  train$ptid <- NULL
  test <- dt[dt$ptid %in% test_ids$V1, ]
  # test$response <- factor(test$response)
  test$ptid <- NULL
  return(list(train, test))
}


build_sgl_model <- function(train, num_features, n_group) {
  set.seed(1)
  size.group <- num_features
  Y <- train$response
  train$response <- NULL
  # index <- ceiling(1:ncol(train)/size.group)
  index <- rep(c(1:size.group), n_group)
  X <- data.matrix(train)
  data <- list(x = X, y = Y)
  # a <- 0.8
  # sglfit <- SGL(data, index, type = "logit", standardize = T, maxit = 800, alpha = a, nlam = 10)
  # out1 <- paste('./data/sgl_updated_model_alpha', as.character(a * 10), '.RData', sep = '')
  # save(sglfit, file = out1)
  # # coefs <- data.frame(sglfit$beta)
  # # out2 <- paste('./data/sgl_coefs_alpha', as.character(a * 10), '.csv', sep = '')
  # # write.csv(coefs, file = out2)

  alphas <- seq(from = 0, to = 1, by = 0.1)
  for (i in 1:length(alphas)) {
    a <- alphas[i]
    print(a)
    sglfit <- SGL(data, index, type = "logit", standardize = T, maxit = 1000, alpha = a, nlam = 15)
    out1 <- paste('./data/sgl_model_', as.character(n_group), 'group_alpha', as.character(a * 10), '_v2.RData', sep = '')
    save(sglfit, file = out1)
    coefs <- data.frame(sglfit$beta)
    out2 <- paste('./data/sgl_coefs_', as.character(n_group), 'group_alpha', as.character(a * 10), '_v2.csv', sep = '')
    write.csv(coefs, file = out2)
  }
  # return(sglfit)
}

  
build_logistic_regression <- function() {
  clf <- glm(response ~., data=train, family = binomial)
  y_pred <- predict(clf, test, type = 'response')
  rocobj <- roc(y_test, y_pred)
  rocobj 
  ci.se(rocobj, boot.n=100, conf.level=0.95, stratified=FALSE)
}

evaluate_performance <- function(clf, test, y_test) {
  y_pred <- predict(clf, test, type = 'response')
  rocobj <- roc(y_test, y_pred)
  recall <- ci.se(rocobj, boot.n=50, conf.level=0.95, stratified=FALSE)
  return(list(rocobj, recall))
}

evaluate_performance_sgl <- function(clf, test, y_test, l) {
  y_pred <- predictSGL(clf, test, lam = l)
  rocobj <- roc(y_test, y_pred)
  print(rocobj)
  # cis <- ci.auc(rocobj, boot.n=50, conf.level=0.95, stratified=FALSE)
  # recall <- ci.se(rocobj, boot.n=50, conf.level=0.95, stratified=FALSE)
  return(rocobj)
  # return(rocobj)
}


# 
# #====================== evaluate baselines ==========================================
dt <- read.csv('./data/test_proba_v5.csv')
# baselin 1 - freq
# threshold tuned by F2 score: RF: 0.47; Lasso: 0.11
rocobj1a <- roc(dt$b1_response, dt$b1_rf) # 0.5992
ci.auc(rocobj1a, conf.level = 0.95, method = 'bootstrap', boot.n = 100)
ci.thresholds(rocobj1a, conf.level = 0.95, method = 'bootstrap', boot.n = 50, thresholds = 0.47)
rocobj1b <- roc(dt$b1_response, dt$b1_gbt) # 0.670
rocobj1b
ci.auc(rocobj1b, conf.level = 0.95, method = 'bootstrap', boot.n = 50)
ci.thresholds(rocobj1b, conf.level = 0.95, method = 'bootstrap', boot.n = 50, thresholds = 0.11)
rocobj1c <- roc(dt$b1_response, dt$b1_lasso) # 0.6398
ci.auc(rocobj1c, conf.level = 0.95, method = 'bootstrap', boot.n = 50)
ci.thresholds(rocobj1c, conf.level = 0.95, method = 'bootstrap', boot.n = 50, thresholds = 0.11)

# baselin 2 - subw sgl
rocobj2a <- roc(dt$b2_response, dt$b2_rf) # 0.8433
rocobj2b <- roc(dt$b2_response, dt$b2_lr) # 0.8644
rocobj2c <- roc(dt$b2_response, dt$b2_lasso) # 0.7843
ci.auc(rocobj2c, conf.level = 0.95, method = 'bootstrap', boot.n = 50)
ci.thresholds(rocobj2a, conf.level = 0.95, method = 'bootstrap', boot.n = 50, thresholds = 0.47)
ci.thresholds(rocobj2c, conf.level = 0.95, method = 'bootstrap', boot.n = 50, thresholds = 0.11)

# baselin 3 - bps
# threshold tuned by F2 score: RF: 0.47; Lasso: 0.11
rocobj3a <- roc(dt$b3_response, dt$b3_rf) # 0.6122
rocobj3b <- roc(dt$b3_response, dt$b3_lr) # 0.6635
rocobj3c <- roc(dt$b3_response, dt$b3_lasso) # 0.6371
ci.auc(rocobj3a, conf.level = 0.95, method = 'bootstrap', boot.n = 50)
ci.thresholds(rocobj3a, conf.level = 0.95, method = 'bootstrap', boot.n = 50, thresholds = 0.47)
ci.thresholds(rocobj3c, conf.level = 0.95, method = 'bootstrap', boot.n = 50, thresholds = 0.11)

rocobj3b <- roc(dt$b3_response, dt$b3_gbt) # 0.699
rocobj3b
ci.auc(rocobj3b, conf.level = 0.95, method = 'bootstrap', boot.n = 50)
ci.thresholds(rocobj3b, conf.level = 0.95, method = 'bootstrap', boot.n = 50, thresholds = 0.11)


# baselin 4 - transition counts
# threshold tuned by F2 score: RF: 0.47; Lasso: 0.11
rocobj4a <- roc(dt$b4_response, dt$b4_rf) # 0.5585
rocobj4b <- roc(dt$b4_response, dt$b4_lr) # 0.5702
rocobj4c <- roc(dt$b4_response, dt$b4_lasso) # 0.5669
ci.auc(rocobj4a, conf.level = 0.95, method = 'bootstrap', boot.n = 50)
ci.thresholds(rocobj4a, conf.level = 0.95, method = 'bootstrap', boot.n = 50, thresholds = 0.51)
ci.thresholds(rocobj4c, conf.level = 0.95, method = 'bootstrap', boot.n = 50, thresholds = 0.11)

rocobj4b <- roc(dt$b4_response, dt$b4_gbt) # 0.656
rocobj4b
ci.auc(rocobj4b, conf.level = 0.95, method = 'bootstrap', boot.n = 50)
ci.thresholds(rocobj4b, conf.level = 0.95, method = 'bootstrap', boot.n = 50, thresholds = 0.11)


# Proposed
# threshold tuned by F2 score: RF: 0.47; Lasso: 0.11
rocobj5a <- roc(dt$b5_response, dt$b5_rf) # 0.8023
rocobj5b <- roc(dt$b5_response, dt$b5_lr) # 0.8216
rocobj5c <- roc(dt$b5_response, dt$b5_lasso) # 0.8067
ci.auc(rocobj5a, conf.level = 0.95, method = 'bootstrap', boot.n = 50)
ci.thresholds(rocobj5a, conf.level = 0.95, method = 'bootstrap', boot.n = 50, thresholds = 0.41)
ci.thresholds(rocobj5c, conf.level = 0.95, method = 'bootstrap', boot.n = 50, thresholds = 0.15)

rocobj5b <- roc(dt$b5_response, dt$b5_gbt) # 0.6638
rocobj5b
ci.auc(rocobj5b, conf.level = 0.95, method = 'bootstrap', boot.n = 50)
ci.thresholds(rocobj5b, conf.level = 0.95, method = 'bootstrap', boot.n = 50, thresholds = 0.11)

# ============== get example patient==============
dt <- read.csv('./data/test_proba_v3.csv')
dt$ptid <- test_ids
write.csv(dt, file = './data/dm_prediction_test.csv', row.names = F)

# # ====================== Baseline 1: aggregated frequency ============================
# file1 <- './data/dm_control_counts.csv'
# data1 <- read_data(file1, 'freq')
# train1 <- data1[[1]]
# test1 <- data1[[2]]
# # logistic regresison
# lr1 <- glm(response ~., data=train1, family = binomial)
# results1a <- evaluate_performance(lr1, test1, test1$response)
# # lasso
# lasso1 <- glmnet(x = data.matrix(train1[, 1:(ncol(train1)-1)]), y = train1$response, alpha=1, family="binomial", lambda = 0.01)
# results1b <- evaluate_performance(lasso1, data.matrix(test1[, 1:(ncol(test1)-1)]), test1$response)
# # random forest
# rf1 <- randomForest(response ~., data=train1)
# results1c <- evaluate_performance(rf1, test1, test1$response)
# 
# 
# # ====================== Baseline 2: BPS frequency ============================
# file2 <- './data/counts_bps.csv'
# data2 <- read_data(file2, 'bps')
# train2 <- data2[[1]]
# test2 <- data2[[2]]
# # logistic regresison
# lr2 <- glm(response ~., data=train2, family = binomial)
# results2a <- evaluate_performance(lr2, test2, test2$response)
# # lasso
# lasso2 <- glmnet(x = data.matrix(train2[, 1:(ncol(train2)-1)]), y = train2$response, alpha=1, family="binomial", lambda = 0.01)
# results2b <- evaluate_performance(lasso2, data.matrix(test2[, 1:(ncol(test2)-1)]), test2$response)
# 
# 
# # ====================== Baseline 3: sub-window frequency ============================
# file3 <- './data/counts_sub.csv'
# data3 <- read_data(file3, 'subw')
# train3 <- data3[[1]]
# test3 <- data3[[2]]
# # logistic regresison
# lr3 <- glm(response ~., data=train3, family = binomial)
# results3a <- evaluate_performance(lr3, test3, test3$response)
# auc3a <- results3a[[1]] # 0.8644
# rec_spec3a <- results3a[[2]]
# # lasso
# lasso3 <- glmnet(x = data.matrix(train3[, 1:(ncol(train3)-1)]), y = train3$response, alpha=1, family="binomial", lambda = 0.01)
# results3b <- evaluate_performance(lasso3, data.matrix(test3[, 1:(ncol(test3)-1)]), test3$response)
# auc3b <- results3b[[1]]
# rec_spec3b <- results3b[[2]]
# results3b # auc: 0.8272
# 
# # ====================== Baseline 4: transition frequency ============================


# ====================== Proposed: SGL-selected sub-window frequency ============================
file4 <- './data/counts_sub_by3month.csv'
data4 <- read_data(file4, 'subw')
train4 <- data4[[1]]
test4 <- data4[[2]]
# train <- train4
build_sgl_model(train4, 34, 4)

file4 <- './data/counts_sub_by2month.csv'
data4 <- read_data(file4, 'subw')
train4 <- data4[[1]]
test4 <- data4[[2]]
# train <- train4
build_sgl_model(train4, 34, 6)

# file4 <- './data/counts_sub_by4month.csv'
# data4 <- read_data(file4, 'subw')
# train4 <- data4[[1]]
# test4 <- data4[[2]]
# # train <- train4
# build_sgl_model(train4, 34, 3)

file4 <- './data/counts_sub_by6month.csv'
data4 <- read_data(file4, 'subw')
train4 <- data4[[1]]
test4 <- data4[[2]]
# train <- train4
build_sgl_model(train4, 34, 2)

# n_group <- 6
# alphas <- seq(from = 0, to = 1, by = 0.1)
# for (i in 1:length(alphas)) {
#   a <- alphas[i]
#   print(a)
#   out1 <- paste('./data/sgl_model_', as.character(n_group), 'group_alpha', as.character(a * 10), '_v2.RData', sep = '')
#   load(out1)
#   coefs <- data.frame(sglfit$beta)
#   out2 <- paste('./data/sgl_coefs_', as.character(n_group), 'group_alpha', as.character(a * 10), '_v2.csv', sep = '')
#   write.csv(coefs, file = out2)
# }


# # =============== plot the performance by alpha and l2ambda ===================================

a <- 0
print(a)
n_group <- 2
out1 <- paste('./data/sgl_model_', as.character(n_group), 'group_alpha', as.character(a * 10), '_v2.RData', sep = '')
load(out1)
res <- c()
for (l in 1:15) {
  results4 <- evaluate_performance_sgl(sglfit, data.matrix(test4[, 1:(ncol(test4)-1)]), test4$response, l)
  res <- c(res, results4$auc)
}
numf <- data.frame(sglfit$lambdas, res)
numf['alpha'] = a

plot(numf$sglfit.lambdas, numf$res, xlab = 'Lambda', ylab = 'AUC', 'b', xlim = c(0.0002, 0), main = paste('alpha =', a))

par(mfrow=c(3,4))
for (a in seq(0, 1, 0.1)) {
  print(a)
  out1 <- paste('./data/sgl_model_', as.character(n_group), 'group_alpha', as.character(a * 10), '_v2.RData', sep = '')
  load(out1)
  res <- c()
  for (l in 1:15) {
  results4 <- evaluate_performance_sgl(sglfit, data.matrix(test4[, 1:(ncol(test4)-1)]), test4$response, l)
  res <- c(res, results4$auc)
  }
  numf <- data.frame(sglfit$lambdas, res)
  numf['alpha'] = a
  plot(numf$sglfit.lambdas, numf$res, xlab = 'Lambda', ylab = 'AUC', 'b', xlim = c(0.0002, 0), main = paste('alpha =', a))
}

# 
# # =============== plot the num_features by alpha and lambda ===================================
# par(mfrow=c(4,3))
# a <- 0
# print(a)
# out1 <- paste('./data/sgl_model_alpha', as.character(a * 10), '.RData', sep = '')
# load(out1)
# res <- c()
# for (l in 1:15) {
#   results4 <- length(sglfit$beta[sapply(sglfit$beta[,l], function(x) x != 0), l])
#   res <- c(res, results4)
# }
# numf <- data.frame(sglfit$lambdas, res)
# numf['alpha'] = a
# 
# plot(numf$sglfit.lambdas, numf$res, xlab = 'Lambda', ylab = '# selected features', 'b', xlim = c(0.0002, 0), main = paste('alpha =', a))
# 
par(mfrow=c(3,4))
for (a in seq(0, 1, 0.1)) {
  print(a)
  out1 <- paste('./data/sgl_model_', as.character(n_group), 'group_alpha', as.character(a * 10), '_v2.RData', sep = '')
  load(out1)
  res <- c()
  for (l in 1:15) {
    results4 <- length(sglfit$beta[sapply(sglfit$beta[,l], function(x) x != 0), l])
    res <- c(res, results4)
  }
  numf <- data.frame(sglfit$lambdas, res)
  numf['alpha'] = a
  plot(numf$sglfit.lambdas, numf$res, xlab = 'Lambda', ylab = '# selected features', 'b', xlim = c(0.0002, 0), main = paste('alpha =', a))
}

save(sglfit, file = './data/updated_sgl.RData')
sglfit$lambdas
num_features <- c(0, 19, 52, 71, 101, 124, 143, 160, 174, 186, 190, 196, 200, 206, 211)
numf <- data.frame(sglfit$lambdas, num_features)
plot(numf$sglfit.lambdas, numf$num_features, xlab = 'Lambda', ylab = '# selected features', 'b', xlim = c(0.0002, 0))

plot(numf$sglfit.lambdas, numf$num_features, xlab = 'Lambda', ylab = '# selected features', 'b', xlim = c(0.0002, 0))

# 
# aa <- data.frame(dt$b2_response, dt$b2_rf)
# 
# 
# alphas <- seq(from = 0, to = 1, by = 0.1)
# for (i in 1:length(alphas)) {
#   a <- alphas[i]
#   print(a)
#   out1 <- paste('./data/sgl_model_', as.character(n_group), 'group_alpha', as.character(a * 10), '_v2.RData', sep = '')
#   load(out1)
#   coefs <- data.frame(sglfit$beta)
#   out2 <- paste('./data/sgl_coefs_', as.character(n_group), 'group_alpha', as.character(a * 10), '_v2.csv', sep = '')
#   write.csv(coefs, file = out2)
# }

