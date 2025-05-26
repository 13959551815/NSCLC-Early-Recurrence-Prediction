# 模型机器---R语言tidymodels包机器学习分类与回归模型---二分类---模型比较

#############################################################
# remotes::install_github("tidymodels/probably")
library(tidymodels)

# 加载各个模型的评估结果
evalfiles <- list.files(".\\cls2\\", full.names = T)
lapply(evalfiles, load, .GlobalEnv)

# 横向比较的模型个数
nmodels <- 10
cols4model <- rainbow(nmodels)  # 模型统一配色
#############################################################

# 各个模型在测试集上的性能指标
predtest_dt$metrics
eval <- bind_rows(
  lapply(list(predtest_logistic, predtest_dt, predtest_enet,
              predtest_knn, predtest_lightgbm, predtest_rf,
              predtest_xgboost, predtest_svm, predtest_mlp,
              predtest_stack), 
         "[[", 
         "metrics")
) %>%
  mutate(model = forcats::as_factor(model))
eval
# 平行线图
eval_max <-   eval %>% 
  group_by(.metric) %>%
  slice_max(.estimate)
eval_min <-   eval %>% 
  group_by(.metric) %>%
  slice_min(.estimate)

eval %>%
  ggplot(aes(x = .metric, y = .estimate, color = model)) +
  geom_point() +
  geom_line(aes(group = model)) +
  ggrepel::geom_text_repel(eval_max,
                           mapping = aes(label = model),
                           nudge_y = 0.05,
                           angle = 90,
                           show.legend = F) +
  ggrepel::geom_text_repel(eval_min,
                           mapping = aes(label = model),
                           nudge_y = -0.05,
                           angle = 90,
                           show.legend = F) +
  scale_color_manual(values = cols4model) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))#10模型的性能指标图86大小

ggsave("x测试集性能平行线图.tiff", plot = last_plot(), width = 7, height = 4, dpi = 300, units = "in")#可以保存最后一张图片

# 指标热图
eval %>%
  select(model, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  mutate(model = reorder(model, roc_auc)) %>%
  pivot_longer(cols = -1) %>%
  group_by(name) %>%
  mutate(valuescale = (value-min(value)) / (max(value)-min(value))) %>%
  ungroup() %>%
  ggplot(aes(x = name, y = model, fill = valuescale)) +
  geom_tile(color = "white", show.legend = F) +
  geom_text(aes(label = round(value, 2))) +
  scale_fill_gradient(low = "green", high = "red") +
  labs(x = "", y = "", fill = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, 
                                   vjust = 1,
                                   hjust = 1))#10模型的性能指标表86大小
ggsave("a测试集性能热图.tiff", plot = last_plot(), width = 7, height = 4, dpi = 300, units = "in")#可以保存最后一张图片

# 各个模型在测试集上的性能指标
predtrain_dt$metrics
eval <- bind_rows(
  lapply(list(predtrain_logistic, predtrain_dt, predtrain_enet,
              predtrain_knn, predtrain_lightgbm, predtrain_rf,
              predtrain_xgboost, predtrain_svm, predtrain_mlp,
              predtrain_stack), 
         "[[", 
         "metrics")
) %>%
  mutate(model = forcats::as_factor(model))
eval
# 平行线图
eval_max <-   eval %>% 
  group_by(.metric) %>%
  slice_max(.estimate)
eval_min <-   eval %>% 
  group_by(.metric) %>%
  slice_min(.estimate)

eval %>%
  ggplot(aes(x = .metric, y = .estimate, color = model)) +
  geom_point() +
  geom_line(aes(group = model)) +
  # ggrepel::geom_text_repel(eval_max, 
  #                          mapping = aes(label = model), 
  #                          nudge_y = 0.05,
  #                          angle = 90,
  #                          show.legend = F) +
  # ggrepel::geom_text_repel(eval_min, 
  #                          mapping = aes(label = model), 
  #                          nudge_y = -0.05,
  #                          angle = 90,
  #                          show.legend = F) +
  scale_color_manual(values = cols4model) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))#10模型在训练集的性能指标图86大小
ggsave("b训练集性能平行线图.tiff", plot = last_plot(), width = 7, height = 4, dpi = 300, units = "in")#可以保存最后一张图片

# 指标热图
eval %>%
  select(model, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  mutate(model = reorder(model, roc_auc)) %>%
  pivot_longer(cols = -1) %>%
  group_by(name) %>%
  mutate(valuescale = (value-min(value)) / (max(value)-min(value))) %>%
  ungroup() %>%
  ggplot(aes(x = name, y = model, fill = valuescale)) +
  geom_tile(color = "white", show.legend = F) +
  geom_text(aes(label = round(value, 2))) +
  scale_fill_gradient(low = "green", high = "red") +
  labs(x = "", y = "", fill = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, 
                                   vjust = 1,
                                   hjust = 1))#10模型在训练集的性能指标表86大小
ggsave("b训练集性能热图.tiff", plot = last_plot(), width = 7, height = 4, dpi = 300, units = "in")#可以保存最后一张图片

# 各个模型在测试集上的性能指标表格
eval2 <- eval %>%
  select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)
eval2

library(writexl)
write_xlsx(eval2, "10模型性能指标汇总250310.xlsx")
# 各个模型在测试集上的性能指标图示
# ROCAUC
eval2 %>%
  ggplot(aes(x = model, y = roc_auc, fill = model)) +
  geom_col(width = 0.5, show.legend = F) +
  geom_text(aes(label = round(roc_auc, 2)), 
            nudge_y = -0.03) +
  scale_fill_manual(values = cols4model) +
  theme_bw()
ggsave("c训练集性能AUC图.tiff", plot = last_plot(), width = 6.5, height = 3, dpi = 300, units = "in")#可以保存最后一张图片

#############################################################

# 各个模型在测试集上的预测概率
predtest <- bind_rows(
  lapply(list(predtest_logistic, predtest_dt, predtest_enet,
              predtest_knn, predtest_lightgbm, predtest_rf,
              predtest_xgboost, predtest_svm, predtest_mlp,
              predtest_stack), 
         "[[", 
         "prediction")
) %>%
  mutate(model = forcats::as_factor(model))
predtest

# 各个模型在测试集上的ROC
predtest %>%
  group_by(model) %>%
  roc_curve(.obs, .pred_1, event_level = "second") %>%
  left_join(eval2[, c("model", "roc_auc")]) %>%
  mutate(modelauc = paste0(model, 
                           ", ROCAUC=", round(roc_auc, 4)),
         modelauc = forcats::as_factor(modelauc)) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = modelauc)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_color_manual(values = cols4model) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  labs(color = "", title = paste0("ROCs on testdata")) +
  theme_bw() +
  theme(legend.position = c(1,0),
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank())
ggsave("d测试集性能ROC曲线图.tiff", plot = last_plot(), width = 5, height = 5, dpi = 300, units = "in")#可以保存最后一张图片

# 各个模型在测试集上的PR
predtest %>%
  group_by(model) %>%
  pr_curve(.obs, .pred_1, event_level = "second") %>%
  left_join(eval2[, c("model", "pr_auc")]) %>%
  mutate(modelauc = paste0(model, 
                           ", PRAUC=", round(pr_auc, 4)),
         modelauc = forcats::as_factor(modelauc)) %>%
  ggplot(aes(x = recall, y = precision, color = modelauc)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_color_manual(values = cols4model) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  labs(color = "", title = paste0("PRs on testdata")) +
  theme_bw() +
  theme(legend.position = c(1, 1.05),#逗号后面的数字是调高低的
        legend.justification = c(1, 1),
        legend.background = element_blank(),
        legend.key = element_blank())#10模型PRAOC曲线汇总图5大小
ggsave("e测试集性能PRAUC曲线图.tiff", plot = last_plot(), width = 5, height = 5, dpi = 300, units = "in")#可以保存最后一张图片

#############################################################

# 各个模型在测试集上的预测概率---宽数据
predtest2 <- predtest %>%
  select(-.pred_0) %>%
  mutate(id = rep(1:nrow(predtest_logistic$prediction), 
                  length(unique(predtest$model)))) %>%
  pivot_wider(id_cols = c(id, .obs), 
              names_from = model, 
              values_from = .pred_1) %>%
  select(id, .obs, sort(unique(predtest$model)))
predtest2

#############################################################


# 各个模型在测试集上的校准曲线
# 校准曲线附加置信区间
library(probably)
# install.packages("probably")
# 算法1
predtest %>%
  cal_plot_breaks(.obs, 
                  .pred_1, 
                  event_level = "second", 
                  num_breaks = 7,  # 可以改大改小
                  .by = model) +
  scale_color_manual(values = cols4model) +
  theme_bw() +
  theme(legend.position = "none")##10模型校准曲线汇总图5大小
# 生成图形
p <- predtest %>%
  cal_plot_windowed(.obs, 
                    .pred_1, 
                    event_level = "second", 
                    window_size = 0.5,  # 可以改大改小
                    .by = model) +
  scale_color_manual(values = cols4model) +
  theme_bw()
p
# 使用 ggsave 保存图形
ggsave("7测试集性能PRAUC曲线图.tiff", plot = p, width = 5, height = 5, dpi = 300, units = "in")

# brier_score
bs <- predtest %>%
  group_by(model) %>%
  yardstick::brier_class(.obs, .pred_0) %>%
  mutate(meanpred = 0.8,
         meanobs = 0.25,
         text = paste0("BS: ", round(.estimate, 3)))
# 附加bs
predtest %>%
  cal_plot_windowed(.obs, 
                    .pred_1, 
                    event_level = "second", 
                    window_size = 0.5,# 可以改大改小
                    .by = model) +
  geom_text(
    bs,
    mapping = aes(x = meanpred - 0.45, y = meanobs + 0.6, label = text)
  ) +
  scale_color_manual(values = cols4model) +
  theme_bw() +
  theme(legend.position = "none") #10模型brier分数汇总图5大小

ggsave("f测试集性能校准曲线图.tiff", plot = last_plot(), width = 5, height = 5, dpi = 300, units = "in")#可以保存最后一张图片


#############################################################

# 各个模型在测试集上的DCA
dca_obj <- dcurves::dca(as.formula(
  paste0(".obs ~ ", 
         paste(colnames(predtest2)[3:ncol(predtest2)], 
               collapse = " + "))
),
  data = predtest2,
  thresholds = seq(0, 1, by = 0.01)
)
plot(dca_obj, smooth = T, span = 0.5) +
  scale_color_manual(values = c("black", "grey", cols4model)) +
  labs(title = "DCA on testdata")#10模型DCA曲线汇总图5大小
ggsave("g测试集性能DCA曲线图.tiff", plot = last_plot(), width = 6, height = 4, dpi = 300, units = "in")#可以保存最后一张图片

#############################################################

# 各个模型交叉验证的各折指标点线图
evalcv <- bind_rows(
  lapply(list(evalcv_logistic, evalcv_dt, evalcv_enet,
              evalcv_knn, evalcv_lightgbm, evalcv_rf,
              evalcv_xgboost, evalcv_svm, evalcv_mlp), 
         "[[", 
         "evalcv")
) %>%
  mutate(
    model = forcats::as_factor(model),
    modelperf = paste0(model, "(", round(mean, 2),"±",
                       round(sd,2), ")")
  )
evalcv

evalcv_max <-   evalcv %>% 
  group_by(id) %>%
  slice_max(.estimate)
evalcv_min <-   evalcv %>% 
  group_by(id) %>%
  slice_min(.estimate)

evalcv %>%
  ggplot(aes(x = id, y = .estimate, 
             group = modelperf, color = modelperf)) +
  geom_point() +
  geom_line() +
  ggrepel::geom_text_repel(evalcv_max, 
                           mapping = aes(label = model), 
                           nudge_y = 0.01,
                           show.legend = F) +
  ggrepel::geom_text_repel(evalcv_min, 
                           mapping = aes(label = model), 
                           nudge_y = -0.01,
                           show.legend = F) +
  scale_y_continuous(limits = c(0.5, 1)) +
  scale_color_manual(values = cols4model) +
  labs(x = "", y = "ROCAUC", color = "Model") +
  theme_bw()#10模型交叉验证的各折指标点线图5乘

# 各个模型交叉验证的指标平均值图(带上下限)
evalcv %>%
  group_by(model) %>%
  sample_n(size = 1) %>%
  ungroup() %>%
  ggplot(aes(x = model, y = mean, color = model)) +
  geom_point(size = 2, show.legend = F) +
  # geom_line(group = 1) +
  geom_errorbar(aes(ymin = mean-sd, 
                    ymax = mean+sd),
                width = 0.1, 
                linewidth = 1.2,
                show.legend = F) +
  scale_y_continuous(limits = c(0.7, 1)) +
  scale_color_manual(values = cols4model) +
  labs(y = "cv roc_auc") +
  theme_bw()#10模型交叉验证的指标平均值图(带上下限)




# 各个模型在训练集上的预测概率
predtrain <- bind_rows(
  lapply(list(predtrain_logistic, predtrain_dt, predtrain_enet,
              predtrain_knn, predtrain_lightgbm, predtrain_rf,
              predtrain_xgboost, predtrain_svm, predtrain_mlp,
              predtrain_stack), 
         "[[", 
         "prediction")
) %>%
  mutate(model = forcats::as_factor(model))
predtrain

# 各个模型在训练集上的ROC
predtrain %>%
  group_by(model) %>%
  roc_curve(.obs, .pred_1, event_level = "second") %>%
  left_join(eval2[, c("model", "roc_auc")]) %>%
  mutate(modelauc = paste0(model, 
                           ", ROCAUC=", round(roc_auc, 4)),
         modelauc = forcats::as_factor(modelauc)) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = modelauc)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_color_manual(values = cols4model) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  labs(color = "", title = paste0("ROCs on traindata")) +
  theme_bw() +
  theme(legend.position = c(1,0),
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank())
ggsave("h训练集性能ROC曲线图.tiff", plot = last_plot(), width = 5, height = 5, dpi = 300, units = "in")#可以保存最后一张图片

# 各个模型在训练集上的PR
predtrain %>%
  group_by(model) %>%
  pr_curve(.obs, .pred_1, event_level = "second") %>%
  left_join(eval2[, c("model", "pr_auc")]) %>%
  mutate(modelauc = paste0(model, 
                           ", PRAUC=", round(pr_auc, 4)),
         modelauc = forcats::as_factor(modelauc)) %>%
  ggplot(aes(x = recall, y = precision, color = modelauc)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_color_manual(values = cols4model) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  labs(color = "", title = paste0("PRs on traindata")) +
  theme_bw() +
  theme(legend.position = c(1, 1.05),#逗号后面的数字是调高低的
        legend.justification = c(1, 1),
        legend.background = element_blank(),
        legend.key = element_blank())#10模型PRAOC曲线汇总图5大小
ggsave("i训练集性能PRAUC曲线图.tiff", plot = last_plot(), width = 5, height = 5, dpi = 300, units = "in")#可以保存最后一张图片

#############################################################

# 各个模型在训练集上的预测概率---宽数据
predtrain2 <- predtrain %>%
  select(-.pred_0) %>%
  mutate(id = rep(1:nrow(predtrain_logistic$prediction), 
                  length(unique(predtrain$model)))) %>%
  pivot_wider(id_cols = c(id, .obs), 
              names_from = model, 
              values_from = .pred_1) %>%
  select(id, .obs, sort(unique(predtrain$model)))
predtrain2

#############################################################


# 各个模型在训练集上的校准曲线
# 校准曲线附加置信区间
library(probably)
# 算法1
predtrain %>%
  cal_plot_breaks(.obs, 
                  .pred_1, 
                  event_level = "second", 
                  num_breaks = 7,  # 可以改大改小
                  .by = model) +
  scale_color_manual(values = cols4model) +
  theme_bw() +
  theme(legend.position = "none")##10模型校准曲线汇总图5大小
# 算法2
predtrain %>%
  cal_plot_windowed(.obs, 
                    .pred_1, 
                    event_level = "second", 
                    window_size = 0.5,  # 可以改大改小
                    .by = model) +
  scale_color_manual(values = cols4model) +
  theme_bw() +
  theme(legend.position = "none")#10模型校准曲线汇总图5大小方法2(训练集)

# brier_score
bs <- predtrain %>%
  group_by(model) %>%
  yardstick::brier_class(.obs, .pred_0) %>%
  mutate(meanpred = 0.8,
         meanobs = 0.25,
         text = paste0("BS: ", round(.estimate, 3)))
# 附加bs
predtrain %>%
  cal_plot_windowed(.obs, 
                    .pred_1, 
                    event_level = "second", 
                    window_size = 0.8,
                    .by = model) +
  geom_text(
    bs,
    mapping = aes(x = meanpred, y = meanobs, label = text)
  ) +
  scale_color_manual(values = cols4model) +
  theme_bw() +
  theme(legend.position = "none")#10模型brier分数汇总图5大小
ggsave("j训练集性能校准曲线图.tiff", plot = last_plot(), width = 5, height = 5, dpi = 300, units = "in")#可以保存最后一张图片



#############################################################

# 各个模型在训练集上的DCA
dca_obj <- dcurves::dca(as.formula(
  paste0(".obs ~ ", 
         paste(colnames(predtrain2)[3:ncol(predtrain2)], 
               collapse = " + "))
),
data = predtrain2,
thresholds = seq(0, 1, by = 0.01)
)
plot(dca_obj, smooth = T, span = 0.5) +
  scale_color_manual(values = c("black", "grey", cols4model)) +
  labs(title = "DCA on traindata")#10模型DCA曲线汇总图5大小
ggsave("k训练集性能DCA曲线图.tiff", plot = last_plot(), width = 5, height = 5, dpi = 300, units = "in")#可以保存最后一张图片

#############################################################

# 各个模型交叉验证的各折指标点线图
evalcv <- bind_rows(
  lapply(list(evalcv_logistic, evalcv_dt, evalcv_enet,
              evalcv_knn, evalcv_lightgbm, evalcv_rf,
              evalcv_xgboost, evalcv_svm, evalcv_mlp), 
         "[[", 
         "evalcv")
) %>%
  mutate(
    model = forcats::as_factor(model),
    modelperf = paste0(model, "(", round(mean, 2),"±",
                       round(sd,2), ")")
  )
evalcv

evalcv_max <-   evalcv %>% 
  group_by(id) %>%
  slice_max(.estimate)
evalcv_min <-   evalcv %>% 
  group_by(id) %>%
  slice_min(.estimate)

evalcv %>%
  ggplot(aes(x = id, y = .estimate, 
             group = modelperf, color = modelperf)) +
  geom_point() +
  geom_line() +
  ggrepel::geom_text_repel(evalcv_max, 
                           mapping = aes(label = model), 
                           nudge_y = 0.01,
                           show.legend = F) +
  ggrepel::geom_text_repel(evalcv_min, 
                           mapping = aes(label = model), 
                           nudge_y = -0.01,
                           show.legend = F) +
  scale_y_continuous(limits = c(0.5, 1)) +
  scale_color_manual(values = cols4model) +
  labs(x = "", y = "ROCAUC", color = "Model") +
  theme_bw()#10模型交叉验证的各折指标点线图5乘

# 各个模型交叉验证的指标平均值图(带上下限)
evalcv %>%
  group_by(model) %>%
  sample_n(size = 1) %>%
  ungroup() %>%
  ggplot(aes(x = model, y = mean, color = model)) +
  geom_point(size = 2, show.legend = F) +
  # geom_line(group = 1) +
  geom_errorbar(aes(ymin = mean-sd, 
                    ymax = mean+sd),
                width = 0.1, 
                linewidth = 1.2,
                show.legend = F) +
  scale_y_continuous(limits = c(0.7, 1)) +
  scale_color_manual(values = cols4model) +
  labs(y = "cv roc_auc") +
  theme_bw()#10模型交叉验证的指标平均值图(带上下限)
