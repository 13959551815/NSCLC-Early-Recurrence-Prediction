# 数据预处理配方
datarecipe_logistic <- recipe(Early_recurrence ~ ., traindata)
datarecipe_logistic

# 设定模型
model_logistic <- logistic_reg(
  mode = "classification",
  engine = "glm"
)
model_logistic

# workflow
wk_logistic <- 
  workflow() %>%
  add_recipe(datarecipe_logistic) %>%
  add_model(model_logistic)
wk_logistic

# 训练模型
set.seed(42)
final_logistic <- wk_logistic %>%
  fit(traindata)
# 检查重新读取的模型
print(final_logistic)
##################################################################

# 训练集预测评估
predtrain_logistic <- eval4cls2(
  model = final_logistic, 
  dataset = traindata, 
  yname = "Early_recurrence", 
  modelname = "Logistic", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_logistic$prediction
predtrain_logistic$predprobplot
predtrain_logistic$predprobplot_fill
predtrain_logistic$rocresult
predtrain_logistic$rocplot
predtrain_logistic$prresult
predtrain_logistic$prplot
predtrain_logistic$cmresult
predtrain_logistic$cmplot
predtrain_logistic$metrics
predtrain_logistic$diycutoff
predtrain_logistic$ksplot
predtrain_logistic$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtrain_logistic$proc)
pROC::ci.auc(predtrain_logistic$proc)

# 预测评估测试集预测评估
predtest_logistic <- eval4cls2(
  model = final_logistic, 
  dataset = testdata, 
  yname = "Early_recurrence", 
  modelname = "Logistic", 
  datasetname = "testdata",
  cutoff = predtrain_logistic$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_logistic$prediction
predtest_logistic$predprobplot
predtest_logistic$predprobplot_fill
predtest_logistic$rocresult
predtest_logistic$rocplot
predtest_logistic$prresult
predtest_logistic$prplot
predtest_logistic$cmresult
predtest_logistic$cmplot
predtest_logistic$metrics
predtest_logistic$diycutoff
predtest_logistic$ksplot
predtest_logistic$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtest_logistic$proc)
pROC::ci.auc(predtest_logistic$proc)

# ROC比较检验
pROC::roc.test(predtrain_logistic$proc, predtest_logistic$proc)

# 合并训练集和测试集上ROC曲线
predtrain_logistic$rocresult %>%
  bind_rows(predtest_logistic$rocresult) %>%
  mutate(dataAUC = paste(data, " ROCAUC:", round(ROCAUC, 4)),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0)) +
  labs(color = "") +
  theme_bw() +
  theme(legend.position = c(1,0),
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank())

# 合并训练集和测试集上PR曲线
predtrain_logistic$prresult %>%
  bind_rows(predtest_logistic$prresult) %>%
  mutate(dataAUC = paste(data, " PRAUC:", round(PRAUC, 4)),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1)) +
  labs(color = "") +
  theme_bw() +
  theme(legend.position = c(0,0),
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank())

# 合并训练集和测试集上性能指标
predtrain_logistic$metrics %>%
  bind_rows(predtest_logistic$metrics) %>%
  select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)
# 合并训练集和测试集上性能指标
performance_metrics <- predtrain_logistic$metrics %>%
  bind_rows(predtest_logistic$metrics) %>%
  select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)

##################################################################

# 交叉验证
set.seed(42)
cv_logistic <- 
  wk_logistic %>%
  fit_resamples(
    folds,
    metrics = metric_set(yardstick::accuracy, 
                         yardstick::roc_auc, 
                         yardstick::pr_auc),
    control = control_resamples(save_pred = T,
                                verbose = T,
                                event_level = "second",
                                parallel_over = "everything",
                                save_workflow = T)
  )
cv_logistic

# 交叉验证指标结果
evalcv_logistic <- collect_predictions(cv_logistic) %>%
  group_by(id) %>%
  roc_auc(Early_recurrence, .pred_1, event_level = "second") %>%
  mutate(model = "logistic",
         mean = mean(.estimate),
         sd = sd(.estimate)/sqrt(length(folds$splits)))
evalcv_logistic

# 交叉验证预测结果图示
collect_predictions(cv_logistic) %>%
  group_by(id) %>%
  roc_curve(Early_recurrence, .pred_1, event_level = "second") %>%
  ungroup() %>%
  left_join(evalcv_logistic, by = "id") %>%
  mutate(idAUC = paste(id, " AUC:", round(.estimate, 4)),
         idAUC = forcats::as_factor(idAUC)) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = idAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  labs(color = "") +
  theme_bw() +
  theme(legend.position = c(1,0),
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank())



##################################################################

# 保存评估结果
save(datarecipe_logistic,
     model_logistic,
     wk_logistic,
     cv_logistic,
     predtrain_logistic,
     predtest_logistic,
     evalcv_logistic,
     file = ".\\cls2\\evalresult_logistic.RData")