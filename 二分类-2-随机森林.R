# 数据预处理配方
datarecipe_rf <- recipe(Early_recurrence ~ ., traindata)
datarecipe_rf

# 设定模型
model_rf <- rand_forest(
  mode = "classification",
  engine = "randomForest", # ranger
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_args(importance = T)
model_rf

# workflow
wk_rf <- 
  workflow() %>%
  add_recipe(datarecipe_rf) %>%
  add_model(model_rf)
wk_rf

#########################  超参数寻优2-贝叶斯优化

# 更新超参数范围
param_rf <- model_rf %>%
  extract_parameter_set_dials() %>%
  update(mtry = mtry(c(2, 10)))

# 贝叶斯优化超参数
set.seed(42)
tune_rf <- wk_rf %>%
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 50,
    param_info = param_rf,
    metrics = metric_set(yardstick::roc_auc,
                         yardstick::accuracy, 
                         yardstick::pr_auc),
    control = control_bayes(save_pred = T, 
                            verbose = T,
                            no_improve = 10,
                            event_level = "second",
                            parallel_over = "everything",
                            save_workflow = T)
  )

########################  超参数寻优结束


# 交叉验证结果
eval_tune_rf <- tune_rf %>%
  collect_metrics()
eval_tune_rf

# 图示
autoplot(tune_rf)
eval_tune_rf %>% 
  filter(.metric == "roc_auc") %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'mtry', values = ~mtry),
      list(label = 'trees', values = ~trees),
      list(label = 'min_n', values = ~min_n)
    )
  ) %>%
  plotly::layout(title = "RF HPO Guided by AUCROC")

# 经过交叉验证得到的最优超参数
hpbest_rf <- tune_rf %>%
  select_by_one_std_err(metric = "roc_auc", desc(min_n))
hpbest_rf

# 采用最优超参数组合训练最终模型
set.seed(42)
final_rf <- wk_rf %>%
  finalize_workflow(hpbest_rf) %>%
  fit(traindata)
final_rf

# 训练集预测评估
predtrain_rf <- eval4cls2(
  model = final_rf, 
  dataset = traindata, 
  yname = "Early_recurrence", 
  modelname = "RF", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_rf$prediction
predtrain_rf$predprobplot
predtrain_rf$predprobplot_fill
predtrain_rf$rocresult
predtrain_rf$rocplot
predtrain_rf$prresult
predtrain_rf$prplot
predtrain_rf$cmresult
predtrain_rf$cmplot
predtrain_rf$metrics
predtrain_rf$diycutoff
predtrain_rf$ksplot
predtrain_rf$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtrain_rf$proc)
pROC::ci.auc(predtrain_rf$proc)

# 预测评估测试集预测评估
predtest_rf <- eval4cls2(
  model = final_rf, 
  dataset = testdata, 
  yname = "Early_recurrence", 
  modelname = "RF", 
  datasetname = "testdata",
  cutoff = predtrain_rf$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_rf$prediction
predtest_rf$predprobplot
predtest_rf$predprobplot_fill
predtest_rf$rocresult
predtest_rf$rocplot
predtest_rf$prresult
predtest_rf$prplot
predtest_rf$dcaplot
predtest_rf$cmresult
predtest_rf$cmplot
predtest_rf$metrics
predtest_rf$diycutoff
predtest_rf$ksplot
predtest_rf$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtest_rf$proc)
pROC::ci.auc(predtest_rf$proc)

# ROC比较检验
pROC::roc.test(predtrain_rf$proc, predtest_rf$proc)


# 合并训练集和测试集上ROC曲线
predtrain_rf$rocresult %>%
  bind_rows(predtest_rf$rocresult) %>%
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
predtrain_rf$prresult %>%
  bind_rows(predtest_rf$prresult) %>%
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
predtrain_rf$metrics %>%
  bind_rows(predtest_rf$metrics) %>%
  select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)
# 合并训练集和测试集上的性能指标
merged_metrics <- predtrain_rf$metrics %>%
  bind_rows(predtest_rf$metrics) %>%
  select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)

# 最优超参数交叉验证的结果
evalcv_rf <- bestcv4cls2(
  wkflow = wk_rf,
  tuneresult = tune_rf,
  hpbest = hpbest_rf,
  yname = "Early_recurrence",
  modelname = "RF",
  v = 5,
  positivelevel = yourpositivelevel
)
evalcv_rf$plotcv
evalcv_rf$evalcv

# 保存评估结果
save(datarecipe_rf,
     model_rf,
     wk_rf,
     param_rf,   
     tune_rf,
     predtrain_rf,
     predtest_rf,
     evalcv_rf,
     file = ".\\cls2\\evalresult_rf.RData")