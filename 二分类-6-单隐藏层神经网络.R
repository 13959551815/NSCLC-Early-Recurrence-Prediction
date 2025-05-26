library(themis)
datarecipe_mlp <- recipe(Early_recurrence ~ ., traindata) %>%
  step_dummy(all_nominal_predictors(), naming = new_dummy_names) %>%
  step_range(all_predictors()) %>%
  step_smote(Early_recurrence)
datarecipe_mlp

# 设定模型
model_mlp <- mlp(
  mode = "classification",
  engine = "nnet",
  hidden_units = tune(),
  penalty = tune(),
  epochs = tune()
) 
model_mlp

# workflow
wk_mlp <- 
  workflow() %>%
  add_recipe(datarecipe_mlp) %>%
  add_model(model_mlp)
wk_mlp

##############################################################

############################  超参数寻优1-网格搜索

# 超参数寻优网格
set.seed(42)
hpgrid_mlp <- parameters(
  hidden_units(range = c(15, 24)),
  penalty(range = c(-3, 0)),
  epochs(range = c(50, 150))
) %>%
  grid_regular(levels = 3) # 常规网格
# grid_random(size = 5) # 随机网格
# grid_latin_hypercube(size = 10) # 拉丁方网格
# grid_max_entropy(size = 10) # 最大熵网格
hpgrid_mlp
# 网格也可以自己手动生成expand.grid()

# 交叉验证网格搜索过程
set.seed(42)
tune_mlp <- wk_mlp %>%
  tune_grid(
    resamples = folds,
    grid = hpgrid_mlp,
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
# install.packages("NeuralNetTools")
# 贝叶斯优化超参数
set.seed(42)
tune_mlp <- wk_mlp %>%
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
eval_tune_mlp <- tune_mlp %>%
  collect_metrics()
eval_tune_mlp

# 图示
autoplot(tune_mlp)
eval_tune_mlp %>% 
  filter(.metric == "roc_auc") %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'hidden_units', values = ~hidden_units),
      list(label = 'penalty', values = ~penalty),
      list(label = 'epochs', values = ~epochs)
    )
  ) %>%
  plotly::layout(title = "MLP HPO Guided by AUCROC")

# 经过交叉验证得到的最优超参数
hpbest_mlp <- tune_mlp %>%
  select_by_one_std_err(metric = "roc_auc", desc(penalty))
hpbest_mlp

# 采用最优超参数组合训练最终模型
set.seed(42)
final_mlp <- wk_mlp %>%
  finalize_workflow(hpbest_mlp) %>%
  fit(traindata)
##################################################################

# 训练集预测评估
predtrain_mlp <- eval4cls2(
  model = final_mlp, 
  dataset = traindata, 
  yname = "Early_recurrence", 
  modelname = "MLP", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_mlp$prediction
predtrain_mlp$predprobplot
predtrain_mlp$predprobplot_fill
predtrain_mlp$rocresult
predtrain_mlp$rocplot
predtrain_mlp$prresult
predtrain_mlp$prplot
predtrain_mlp$cmresult
predtrain_mlp$cmplot
predtrain_mlp$metrics
predtrain_mlp$diycutoff
predtrain_mlp$ksplot
predtrain_mlp$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtrain_mlp$proc)
pROC::ci.auc(predtrain_mlp$proc)

# 预测评估测试集预测评估
predtest_mlp <- eval4cls2(
  model = final_mlp, 
  dataset = testdata, 
  yname = "Early_recurrence", 
  modelname = "MLP", 
  datasetname = "testdata",
  cutoff = predtrain_mlp$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_mlp$prediction
predtest_mlp$predprobplot
predtest_mlp$predprobplot_fill
predtest_mlp$rocresult
predtest_mlp$rocplot
predtest_mlp$prresult
predtest_mlp$prplot
predtest_mlp$cmresult
predtest_mlp$cmplot
predtest_mlp$metrics
predtest_mlp$diycutoff
predtest_mlp$ksplot
predtest_mlp$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtest_mlp$proc)
pROC::ci.auc(predtest_mlp$proc)

# ROC比较检验
pROC::roc.test(predtrain_mlp$proc, predtest_mlp$proc)

# 合并训练集和测试集上ROC曲线
predtrain_mlp$rocresult %>%
  bind_rows(predtest_mlp$rocresult) %>%
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
predtrain_mlp$prresult %>%
  bind_rows(predtest_mlp$prresult) %>%
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
predtrain_mlp$metrics %>%
  bind_rows(predtest_mlp$metrics) %>%
  select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


table(traindata$Early_recurrence)
table(testdata$Early_recurrence)

# 最优超参数交叉验证的结果
evalcv_mlp <- bestcv4cls2(
  wkflow = wk_mlp,
  tuneresult = tune_mlp,
  hpbest = hpbest_mlp,
  yname = "Early_recurrence",
  modelname = "MLP",
  v = 5,
  positivelevel = yourpositivelevel
)
evalcv_mlp$plotcv
evalcv_mlp$evalcv

# 保存评估结果
save(datarecipe_mlp,
     model_mlp,
     wk_mlp,
     # hpgrid_mlp,
     tune_mlp,
     predtrain_mlp,
     predtest_mlp,
     evalcv_mlp,
     file = ".\\cls2\\evalresult_mlp.RData")