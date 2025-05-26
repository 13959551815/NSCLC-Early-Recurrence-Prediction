
# 数据预处理配方
datarecipe_enet <- recipe(Early_recurrence ~ ., traindata) %>%
  step_dummy(all_nominal_predictors(), 
             naming = new_dummy_names) %>% 
  step_normalize(all_predictors())
datarecipe_enet


# 设定模型
model_enet <- logistic_reg(
  mode = "classification",
  engine = "glmnet",
  # mixture = 1,   # LASSO
  # mixture = 0,  # 岭回归
  mixture = tune(),
  penalty = tune()
)
model_enet

# workflow
wk_enet <- 
  workflow() %>%
  add_recipe(datarecipe_enet) %>%
  add_model(model_enet)
wk_enet

##############################################################

############################  超参数寻优1-网格搜索

# 超参数寻优网格
set.seed(42)
hpgrid_enet <- parameters(
  mixture(),
  penalty(range = c(-5, 0))
) %>%
  grid_regular(levels = c(5, 20)) # 常规网格
# grid_random(size = 5) # 随机网格
# grid_latin_hypercube(size = 10) # 拉丁方网格
# grid_max_entropy(size = 10) # 最大熵网格
hpgrid_enet
# 网格也可以自己手动生成expand.grid()

# 交叉验证网格搜索过程
set.seed(42)
tune_enet <- wk_enet %>%
  tune_grid(
    resamples = folds,
    grid = hpgrid_enet,
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
tune_enet <- wk_enet %>%
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
eval_tune_enet <- tune_enet %>%
  collect_metrics()
eval_tune_enet

# 图示
# autoplot(tune_enet)
eval_tune_enet %>% 
  filter(.metric == "roc_auc") %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'mixture', values = ~mixture),
      list(label = 'penalty', values = ~penalty)
    )
  ) %>%
  plotly::layout(title = "ENet HPO Guided by AUCROC")

# 经过交叉验证得到的最优超参数
hpbest_enet <- tune_enet %>%
  select_by_one_std_err(metric = "roc_auc", desc(penalty))
hpbest_enet

# 采用最优超参数组合训练最终模型
set.seed(42)
final_enet <- wk_enet %>%
  finalize_workflow(hpbest_enet) %>%
  fit(traindata)

##################################################################

# 训练集预测评估
predtrain_enet <- eval4cls2(
  model = final_enet, 
  dataset = traindata, 
  yname = "Early_recurrence", 
  modelname = "ENet", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_enet$prediction
predtrain_enet$predprobplot
predtrain_enet$predprobplot_fill
predtrain_enet$rocresult
predtrain_enet$rocplot
predtrain_enet$prresult
predtrain_enet$prplot
predtrain_enet$cmresult
predtrain_enet$cmplot
predtrain_enet$metrics
predtrain_enet$diycutoff
predtrain_enet$ksplot
predtrain_enet$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtrain_enet$proc)
pROC::ci.auc(predtrain_enet$proc)

# 预测评估测试集预测评估
predtest_enet <- eval4cls2(
  model = final_enet, 
  dataset = testdata, 
  yname = "Early_recurrence", 
  modelname = "ENet", 
  datasetname = "testdata",
  cutoff = predtrain_enet$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_enet$prediction
predtest_enet$predprobplot
predtest_enet$predprobplot_fill
predtest_enet$rocresult
predtest_enet$rocplot
predtest_enet$prresult
predtest_enet$prplot
predtest_enet$cmresult
predtest_enet$cmplot
predtest_enet$metrics
predtest_enet$diycutoff
predtest_enet$ksplot
predtest_enet$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtest_enet$proc)
pROC::ci.auc(predtest_enet$proc)

# ROC比较检验
pROC::roc.test(predtrain_enet$proc, predtest_enet$proc)


# 合并训练集和测试集上ROC曲线
predtrain_enet$rocresult %>%
  bind_rows(predtest_enet$rocresult) %>%
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
predtrain_enet$prresult %>%
  bind_rows(predtest_enet$prresult) %>%
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
predtrain_enet$metrics %>%
  bind_rows(predtest_enet$metrics) %>%
  select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


# 最优超参数交叉验证的结果
evalcv_enet <- bestcv4cls2(
  wkflow = wk_enet,
  tuneresult = tune_enet,
  hpbest = hpbest_enet,
  yname = "Early_recurrence",
  modelname = "ENet",
  v = 5,
  positivelevel = yourpositivelevel
)
evalcv_enet$plotcv
evalcv_enet$evalcv

# 保存评估结果
save(datarecipe_enet,
     model_enet,
     wk_enet,
     # hpgrid_enet,   # 如果采用贝叶斯优化则无需保存
     tune_enet,
     predtrain_enet,
     predtest_enet,
     evalcv_enet,
     file = ".\\cls2\\evalresult_enet.RData")