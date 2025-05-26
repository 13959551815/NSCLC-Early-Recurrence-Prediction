# 数据预处理配方
datarecipe_svm <- recipe(Early_recurrence ~ ., traindata) %>%
  step_dummy(all_nominal_predictors(), 
             naming = new_dummy_names) %>% 
  step_normalize(all_predictors())
datarecipe_svm


# 设定模型
model_svm <- svm_rbf(     # 高斯核(径向基核),可以替换为线性核、多项式核
  mode = "classification",
  engine = "kernlab",
  cost = tune(),
  rbf_sigma = tune()
)
model_svm

# workflow
wk_svm <- 
  workflow() %>%
  add_recipe(datarecipe_svm) %>%
  add_model(model_svm)
wk_svm

##############################################################
############################  超参数寻优1-网格搜索

# 超参数寻优网格
set.seed(42)
hpgrid_svm <- parameters(
  cost(range = c(-5, 5)), 
  rbf_sigma(range = c(-4, -1))
) %>%
  # grid_regular(levels = c(2,3)) # 常规网格
  grid_random(size = 20) # 随机网格
# grid_latin_hypercube(size = 10) # 拉丁方网格
# grid_max_entropy(size = 10) # 最大熵网格
hpgrid_svm
# 网格也可以自己手动生成expand.grid()

# 交叉验证网格搜索过程
set.seed(42)
tune_svm <- wk_svm %>%
  tune_grid(
    resamples = folds,
    grid = hpgrid_svm,
    metrics = metric_set(yardstick::accuracy, 
                         yardstick::roc_auc, 
                         yardstick::pr_auc),
    control = control_grid(save_pred = T, 
                           verbose = T,
                           event_level = "second",
                           parallel_over = "everything",
                           save_workflow = T)
  )

#########################  超参数寻优2-贝叶斯优化

# 贝叶斯优化超参数
set.seed(42)
tune_svm <- wk_svm %>%
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 50,
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
eval_tune_svm <- tune_svm %>%
  collect_metrics()
eval_tune_svm

# 图示
autoplot(tune_svm)
eval_tune_svm %>% 
  filter(.metric == "roc_auc") %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'cost', values = ~cost),
      list(label = 'rbf_sigma', values = ~rbf_sigma)
    )
  ) %>%
  plotly::layout(title = "SVM HPO Guided by AUCROC")

# 经过交叉验证得到的最优超参数
hpbest_svm <- tune_svm %>%
  select_best(metric = "roc_auc")

# 采用最优超参数组合训练最终模型
set.seed(42)
final_svm <- wk_svm %>%
  finalize_workflow(hpbest_svm) %>%
  fit(traindata)
final_svm

##################################################################

# 训练集预测评估
predtrain_svm <- eval4cls2(
  model = final_svm, 
  dataset = traindata, 
  yname = "Early_recurrence", 
  modelname = "SVM", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_svm$prediction
predtrain_svm$predprobplot
predtrain_svm$predprobplot_fill
predtrain_svm$rocresult
predtrain_svm$rocplot
predtrain_svm$prresult
predtrain_svm$prplot
predtrain_svm$cmresult
predtrain_svm$cmplot
predtrain_svm$metrics
predtrain_svm$diycutoff
predtrain_svm$ksplot
predtrain_svm$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtrain_svm$proc)
pROC::ci.auc(predtrain_svm$proc)

# 预测评估测试集预测评估
predtest_svm <- eval4cls2(
  model = final_svm, 
  dataset = testdata, 
  yname = "Early_recurrence", 
  modelname = "SVM", 
  datasetname = "testdata",
  cutoff = predtrain_svm$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_svm$prediction
predtest_svm$predprobplot
predtest_svm$predprobplot_fill
predtest_svm$rocresult
predtest_svm$rocplot
predtest_svm$prresult
predtest_svm$prplot
predtest_svm$cmresult
predtest_svm$cmplot
predtest_svm$metrics
predtest_svm$diycutoff
predtest_svm$ksplot
predtest_svm$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtest_svm$proc)
pROC::ci.auc(predtest_svm$proc)


# ROC比较检验
pROC::roc.test(predtrain_svm$proc, predtest_svm$proc)

# 合并训练集和测试集上ROC曲线
predtrain_svm$rocresult %>%
  bind_rows(predtest_svm$rocresult) %>%
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
predtrain_svm$prresult %>%
  bind_rows(predtest_svm$prresult) %>%
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
predtrain_svm$metrics %>%
  bind_rows(predtest_svm$metrics) %>%
  select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)
# 合并训练集和测试集上的性能指标
merged_metrics_svm <- predtrain_svm$metrics %>%
  bind_rows(predtest_svm$metrics) %>%
  select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)

# 最优超参数交叉验证的结果
evalcv_svm <- bestcv4cls2(
  wkflow = wk_svm,
  tuneresult = tune_svm,
  hpbest = hpbest_svm,
  yname = "Early_recurrence",
  modelname = "SVM",
  v = 5,
  positivelevel = yourpositivelevel
)
evalcv_svm$plotcv
evalcv_svm$evalcv

# 保存评估结果
save(datarecipe_svm,
     model_svm,
     wk_svm,
     # hpgrid_svm,
     tune_svm,
     predtrain_svm,
     predtest_svm,
     evalcv_svm,
     file = ".\\cls2\\evalresult_svm.RData")