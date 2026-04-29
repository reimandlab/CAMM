setwd("~/Downloads")
library("ggplot2")
library("reshape2")
library("patchwork")

dat = read.delim("z2_to_z5_summary.txt", stringsAsFactors = F)
dat$z1 = paste0("z=", dat$z)

dat_melt1 = melt(dat[, c("z1", "selected.windows", "cancer.gene.windows")], id.var = "z1")

plt1 = ggplot(dat_melt1, aes(z1, value, fill = variable)) + 
		geom_bar(stat = "identity", position = "dodge") + 
		scale_fill_manual(values = c("selected.windows" = "darkgrey", "cancer.gene.windows" = "darkred")) + 
		theme_bw()

plt2 = ggplot(dat, aes(z1, log10(P.value))) +
		geom_bar(stat = "identity", width = 0.002, color = "black") +
		geom_point(size = 5, pch = 21, color = "black", aes(fill = odds.ratio))  +
		scale_fill_gradient(low = "white", high = "red", limits = c(0, 12)) + 
		theme_bw()
		
		
plt_combined = plt1 / plt2
plt_combined

ggsave("figure.pdf", width = 5, height = 7)
system("open figure.pdf")