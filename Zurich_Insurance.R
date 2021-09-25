
require(gtools)
require(Hmisc)
require(gplots)
require(LSD)
require(corrplot)
library(igraph)
require(plotrix)
library(RColorBrewer)
library(scales)
library(grid)
library(ggridges)
library(padr)
library(ggiraphExtra)
library(ggradar)
library(extrafont)
font_import()
loadfonts(device = "win")
library(ggplot2)
library(viridis)
library(hrbrthemes)
library(reshape2)
require(drc)
library(ggsci)
library(Hmisc)
library(RColorBrewer)
require(drm)
library(growthcurver)
library(pheatmap)
library(plotly)
library(data.table)
library(gifski)
library(gridExtra)
library(gganimate)
library(dplyr)
library(ggpubr)
library(tibble)
library(data.table)
library(magick)
library(tidyverse)
library(RCircos)
require(gtools)
require(graphics)
require(stats)
require(zoo)
require(corrplot)
library(igraph)
require(plotrix)
library(matrixStats)
require(drc)
library(ggrepel)
library(ggplot2)
library(ggridges)
library(gsubfn) 
library(readr)
library(lubridate)
par(mar=c(1,1,1,1))
create_beautiful_radarchart <- function(data, color = "#00AFBB", 
                                        vlabels = colnames(data), vlcex = 0.7,
                                        caxislabels = NULL, title = NULL, ...){
  radarchart(
    data, axistype = 1,
    # Customize the polygon
    pcol = color, pfcol = scales::alpha(color, 0.5), plwd = 2, plty = 1,
    # Customize the grid
    cglcol = "grey", cglty = 1, cglwd = 0.8,
    # Customize the axis
    axislabcol = "grey", 
    # Variable labels
    vlcex = vlcex, vlabels = vlabels,
    caxislabels = caxislabels, title = title, ...
  )
}
library(fmsb)



move_data <- data.frame(read_csv("/Users/bassler/Desktop/Documents/Hackathon_ZI/daily_summaries.csv"))
sleep_raw <- data.frame(read_csv("/Users/bassler/Desktop/Documents/Hackathon_ZI/sleep_revisited.csv"))
profile_data <- data.frame(read_csv("/Users/bassler/Desktop/Documents/Hackathon_ZI/user_profile.csv"))
user_list <- unique(move_data$userId)

#####Move#####
move_data$date <- ymd(as.character(move_data$date))
move_data$day_mon <- as.yearmon(move_data$date)
move_data$final_activity <- "move"
move_data$calories <- move_data$calories / 100
move_data$steps <- move_data$steps / 250
move_data$distance <- move_data$distance / 1000
# move_data$calories <- move_data$calories - median(move_data$calories)
# move_data$steps <- move_data$steps - median(move_data$steps)
# move_data$distance <- move_data$distance - median(move_data$distance)
move_data <- pivot_longer(move_data, cols = 4:8, names_to = "recognized_activity", values_to = "score")
#write.csv(move_data, "/Users/bassler/Desktop/Documents/Hackathon_ZI/final_move_set.csv")

for (n in 1:length(user_list)){
  test_data <- move_data[move_data$userId == user_list[n] & move_data$recognized_activity != "points" & as.Date(move_data$date) > as.Date("2021-09-01"),]
  #ggplot(test_data, aes(x=as.Date(date), y=recognized_activity, fill = recognized_activity)) + 
  ggplot(test_data, aes(x=as.Date(date), y=score)) + 
    geom_ridgeline(aes(fill=recognized_activity, height = score, colour = recognized_activity))+
    #geom_density_ridges(scale = 0.5,bandwidth = 0.2,rel_min_height = 0.01, colour= NA )+
    scale_x_date(date_labels = '%d-%b', date_breaks  ="3 days")+#, limits = c(as.Date("2020-02-01"), as.Date("2021-03-01")))+
    theme(axis.text.x = element_text(angle=75, vjust=0.6, size=12, family="SF Pro Display"),
          panel.background = element_blank(),#, axis.line = element_line(colour = "black"),
          plot.title = element_blank(),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          axis.text.y=element_blank(),
          axis.ticks.y=element_blank(),
          legend.position = "right",
          legend.title = element_blank()) -> p
  ggsave(paste0("/Users/bassler/Desktop/Documents/Hackathon_ZI/Move_plot/Plots_User_",n,"_move1.png"), plot=p)
  
  ggplot(test_data, aes(x=as.Date(date), y=..density..)) + 
    geom_density(aes(fill=recognized_activity),color=NA, position="stack", adjust = 0.005)+
    #geom_density(aes(fill=recognized_activity),color=NA, position = position_dodge(width = 0.9), adjust = 0.005)+
    scale_x_date(date_labels = '%d-%b', date_breaks  ="3 days")+#, limits = c(as.Date("2020-02-01"), as.Date("2021-03-01")))+
    theme(axis.text.x = element_text(angle=75, vjust=0.6, size=12, family="SF Pro Display"),
          panel.background = element_blank(),#, axis.line = element_line(colour = "black"),
          plot.title = element_blank(),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          axis.text.y=element_blank(),
          axis.ticks.y=element_blank(),
          legend.position = "right",
          legend.title = element_blank()) -> p
  ggsave(paste0("/Users/bassler/Desktop/Documents/Hackathon_ZI/Move_plot/Plots_User_",n,"_move2.png"), plot=p)
}


#Profile data / age
ID <- c()
age <- c()
metabolism <- c()
height_list <- c()
weight_list <- c()
BMI <- c()
BMI_list <- c()
BMI_category <- c()
for (n in user_list){
  ID <- c(ID, n)
  age <- c(age,2021-profile_data[profile_data$id == n,][1,]$yearOfBirth)
  metabolism <- c(metabolism, profile_data[profile_data$id == n,][1,]$basalMetabolism)
  height_list <- c(height_list, profile_data[profile_data$id == n & profile_data$heightUOM == "cm",][1,]$height)
  height <- profile_data[profile_data$id == n & profile_data$heightUOM == "cm",][1,]$height
  weight_list <- c(weight_list, profile_data[profile_data$id == n & profile_data$weightUOM == "kg",][1,]$weight)
  weight <- profile_data[profile_data$id == n & profile_data$weightUOM == "kg",][1,]$weight
  BMI <- weight/(height/100)^2
  BMI_list <- c(BMI_list, weight/(height/100)^2)
  if (BMI < 18.5){
    BMI_category <- c(BMI_category,"Underweight")
  }
  if (18.5 <= BMI & BMI <= 25){
    BMI_category <- c(BMI_category,"Normal weight")
  }
  if (25 < BMI & BMI < 30){
    BMI_category <- c(BMI_category,"Overweight")
  }
  if (30 <= BMI){
    BMI_category <- c(BMI_category,"Obesity")
  }
}
final_profile <- data.frame(ID, age, metabolism, height_list, weight_list, BMI_list, BMI_category)
write.csv(final_profile, "/Users/bassler/Desktop/Documents/Hackathon_ZI/final_profile_set.csv")


#Sleep
sleep_users <- unique(sleep_data$id)
sleep_data <- select(sleep_raw, id, newCalendarDate, durationInSeconds, deepSleepDurationInSeconds,lightSleepDurationInSeconds,remSleepSeconds )
sleep_data$date <- ymd(as.character(sleep_data$newCalendarDate)) 
sleep_data$day_mon <- as.yearmon(sleep_data$date)
sleep_data$duration_hours <- sleep_data$durationInSeconds / 3600
sleep_data$deep_duration_hours <- sleep_data$deepSleepDurationInSeconds / 3600
sleep_data$light_duration_hours <- sleep_data$lightSleepDurationInSeconds / 3600
sleep_data$rem_duration_hours <- sleep_data$remSleepSeconds / 3600
sleep_data$final_activity <- "recharge"
sleep_data$id [1:13] <- user_list [1]
sleep_data$id [14:23] <- user_list [2]
sleep_data$id [24:33] <- user_list [3]
sleep_data$id [55:66] <- user_list [4]
sleep_data$id [68:77] <- user_list [5]
sleep_data$id [78:87] <- user_list [6]
sleep_data$id [109:121] <- user_list [7]
sleep_data$id [143:150] <- user_list [8]
sleep_data$id [c(35,36,38,41,44,48,49,52)] <- user_list [9]


sleep_data <- pivot_longer(sleep_data, cols = 9:12, names_to = "recognized_activity", values_to = "score")
write.csv(sleep_data, "/Users/bassler/Desktop/Documents/Hackathon_ZI/final_sleep_set.csv")

for (n in 1:length(user_list)){
  test_data <- sleep_data_new[sleep_data$id == user_list[n],] #& as.Date(sleep_data$date) > as.Date("2021-09-01"),]
  #ggplot(test_data, aes(x=as.Date(date), y=recognized_activity, fill = recognized_activity)) + 
  ggplot(test_data, aes(x=as.Date(date), y=score)) + 
    geom_ridgeline(aes(fill=recognized_activity, height = score, colour = recognized_activity))+
    #geom_density_ridges(scale = 0.5,bandwidth = 0.2,rel_min_height = 0.01, colour= NA )+
    scale_x_date(date_labels = '%d-%b', date_breaks  ="3 days")+#, limits = c(as.Date("2020-02-01"), as.Date("2021-03-01")))+
    theme(axis.text.x = element_text(angle=75, vjust=0.6, size=12, family="SF Pro Display"),
          panel.background = element_blank(),#, axis.line = element_line(colour = "black"),
          plot.title = element_blank(),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          axis.text.y=element_blank(),
          axis.ticks.y=element_blank(),
          legend.position = "right",
          legend.title = element_blank()) -> p
  ggsave(paste0("/Users/bassler/Desktop/Documents/Hackathon_ZI/Sleep_plot/Plots_User_",n,"_move1.png"), plot=p)
  
  ggplot(test_data, aes(x=as.Date(date), y=..density..)) + 
    geom_density(aes(fill=recognized_activity),color=NA, position="stack", adjust = 0.005)+
    #geom_density(aes(fill=recognized_activity),color=NA, position = position_dodge(width = 0.9), adjust = 0.005)+
    scale_x_date(date_labels = '%d-%b', date_breaks  ="3 days")+#, limits = c(as.Date("2020-02-01"), as.Date("2021-03-01")))+
    theme(axis.text.x = element_text(angle=75, vjust=0.6, size=12, family="SF Pro Display"),
          panel.background = element_blank(),#, axis.line = element_line(colour = "black"),
          plot.title = element_blank(),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          axis.text.y=element_blank(),
          axis.ticks.y=element_blank(),
          legend.position = "right",
          legend.title = element_blank()) -> p
  ggsave(paste0("/Users/bassler/Desktop/Documents/Hackathon_ZI/Sleep_plot/Plots_User_",n,"_move2.png"), plot=p)
}




#Food
hel_data <- read.table("/Users/bassler/Desktop/Documents/Hackathon_Delage/Summary.txt", header = T, sep="\t")
user_list_old <- unique(full_data$ID)
test_data <- hel_data [hel_data$ID == user_list_old[4] & hel_data$final_activity == "eat" & as.Date(hel_data$activity_day) > as.Date("2020-01-01"),]
test_data$activity_day <- c("2021-09-04", "2020-09-09", "2021-09-16", "2021-09-13", "2021-09-09", "2021-09-03", "2021-09-12", "2020-09-04", "2021-09-05", "2021-09-18", "2021-09-03", "2021-09-19", "2021-09-19", "2021-09-21",
                             "2021-09-07", "2021-09-11", "2021-09-10", "2021-09-20", "2021-09-17", "2021-09-25", "2021-09-02", "2021-09-19", "2021-09-14", "2021-09-08", "2021-09-15", "2021-09-24")


test_data <- example [as.Date(example$activity_day) > as.Date("2021-09-01") & as.Date(example$activity_day) < as.Date("2021-09-26"),]
ggplot(test_data, aes(x=as.Date(activity_day), y=..density..)) + 
  geom_density(aes(fill=recognized_activity),color=NA, position="stack", adjust = 0.005)+
  #geom_density(aes(fill=recognized_activity),color=NA, position = position_dodge(width = 0.9), adjust = 0.005)+
  scale_x_date(date_labels = '%d-%b', date_breaks  ="3 days")+#, limits = c(as.Date("2020-02-01"), as.Date("2021-03-01")))+
  theme(axis.text.x = element_text(angle=75, vjust=0.6, size=12, family="SF Pro Display"),
        panel.background = element_blank(),#, axis.line = element_line(colour = "black"),
        plot.title = element_blank(),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        legend.position = "right",
        legend.title = element_blank()) -> p
ggsave(paste0("/Users/bassler/Desktop/Documents/Hackathon_ZI/Eat_plot/Plots_User_",n,"_eat.png"), plot=p)


dur <- median(sleep_data[sleep_data$recognized_activity == "duration_hours",]$score)
deep_dur <- median(sleep_data[sleep_data$recognized_activity == "deep_duration_hours",]$score)
light_dur <- median(sleep_data[sleep_data$recognized_activity == "light_duration_hours",]$score)
rem_dur <- median(sleep_data[sleep_data$recognized_activity == "rem_duration_hours",]$score)

recharge_score <- c()
move_score <- c()
eat_score <- c()
for (n in 1:9){
  udur <- median(sleep_data[sleep_data$recognized_activity == "duration_hours" & sleep_data$id == user_list[n],]$score)
  udeep_dur <- median(sleep_data[sleep_data$recognized_activity == "deep_duration_hours" &sleep_data$id == user_list[n],]$score)
  ulight_dur <- median(sleep_data[sleep_data$recognized_activity == "light_duration_hours" &  sleep_data$id == user_list[n],]$score)
  urem_dur <- median(sleep_data[sleep_data$recognized_activity == "rem_duration_hours" &  sleep_data$id == user_list[n],]$score)
  
  
  
  recharge_score <- c(recharge_score, mean(c((udur/dur) * 100, (udeep_dur/deep_dur) * 100 ,(ulight_dur/light_dur) * 100)))
  move_score <- c(move_score, mean(move_data[move_data$userId == user_list[n] & move_data$recognized_activity == "points",]$score))
  eat_score <- c(eat_score, mean(c(mean(c((udur/dur * 100),(udeep_dur/deep_dur * 100),(ulight_dur/light_dur * 100))), mean(move_data[move_data$userId == user_list[n] & move_data$recognized_activity == "points",]$score))))
}

score_frame <- data_frame(user_list, recharge_score, move_score, eat_score)
max_recharge <- max(score_frame$recharge_score)
max_move <- max(score_frame$move_score)
max_eat <- max(score_frame$eat_score)


#####Total score#####
recharge_score <- c()
move_score <- c()
eat_score <- c()
total_score <- c()
for (n in 1:length(user_list)){
  recharge_score <- c(recharge_score, (((score_frame[score_frame$user_list == user_list[n],]$recharge_score/max_recharge)*100)))
  eat_score <- c(eat_score, (((score_frame[score_frame$user_list == user_list[n],]$eat_score/max_eat)*100)))
  move_score <- c(move_score, (((score_frame[score_frame$user_list == user_list[n],]$move_score/max_move)*100)))
  total_score <- c(total_score, mean(c(
    ((score_frame[score_frame$user_list == user_list[n],]$recharge_score/max_recharge)*100),
    ((score_frame[score_frame$user_list == user_list[n],]$eat_score/max_eat)*100),
    ((score_frame[score_frame$user_list == user_list[n],]$move_score/max_move)*100))))}
    
user_scores <- data.frame (user_list, recharge_score, move_score, eat_score, total_score)
colnames(user_scores) <- c("user_list", "Recharge","Move","Eat","Total")
write.csv(user_scores, paste0("/Users/bassler/Desktop/Documents/Hackathon_ZI/final_user_scores.csv"))


write.csv(final_data, paste0("/Users/bassler/Desktop/Documents/Hackathon_ZI/final_data.csv"))

#####Spider plot radarchart#####
for (n in c(1:9)){
  png(paste0("/Users/bassler/Desktop/Documents/Hackathon_ZI/Total_score/Total_score_",n,"_spider.png"))
  data <-user_scores[n, ]
  rownames(data) <- data$user_list
  data<- data[,2:5]
  max_min <- data.frame(
    Recharge = c(100, 0), Move = c(100, 0), Eat = c(100, 0),
    Total = c(100, 0))
  df <- rbind(max_min, data)
  rownames(df) <- c("Max", "Min", "User")
  
  op <- par(mar = c(1, 1, 1, 1))
  create_beautiful_radarchart(df, caxislabels = c(0, 25, 50, 75, 100))
  par(op)
  dev.off()
}


#####Spider ggradar#####
#colnames(user_scores) <- c("user_list", "Recharge", "Move","Eat","Total")
for (n in c(1:9)){
  ggradar(
    user_scores[n, ], 
    values.radar = c("0", "50", "100"),
    grid.min = 0, grid.mid = 50, grid.max = 100,
    # Polygons
    group.line.width = 1, 
    group.point.size = 3,
    group.colours = "#00AFBB",
    pcol = color, pfcol = scales::alpha(color, 0.5), plwd = 2, plty = 1,
    # Customize the grid
    cglcol = "lightblue", cglty = 1, cglwd = 0.8,
    # Background and grid lines
    background.circle.colour = "lightblue",
    gridline.mid.colour = "grey"
  ) -> p
  ggsave(paste0("/Users/bassler/Desktop/Documents/Hackathon_ZI/Total_score/Plots_User_",n,"_spider.png"), plot=p)
}


#####Spider ggplot#####
scores <- user_scores[n, ]
colnames(scores) <- c("user_list", "Recharge", "Move", "Eat", "Total" )
data <- pivot_longer(scores, cols = c(recharge_score, move_score, eat_score, total_score))

ggplot(data, aes(name, value, group = 1)) +
  geom_polygon(fill = "blue", colour = "blue", alpha = 0.4) +
  geom_point(colour = "blue") +
  coord_radar() +
  ylim(0, 100) +
  theme(axis.text.x = element_text(size=12, family="SF Pro Display"),
        plot.title = element_blank(),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank())
#####Move#####
for (n in good_users){
  test_data <- full_data [full_data$ID %in% user_list[n] & full_data$final_activity == "move",]
  ggplot(test_data, aes(x=as.Date(activity_day), y=..density..)) + 
    geom_density(aes(fill=recognized_activity),color=NA, position="stack")+
    scale_x_date(date_labels = '%b-%Y', date_breaks  ="3 month", limits = c(as.Date("2020-02-01"), as.Date("2021-03-01")))+
    theme(axis.text.x = element_text(angle=75, vjust=0.6, size=12, family="SF Pro Display"),
          panel.background = element_blank(),#, axis.line = element_line(colour = "black"),
          plot.title = element_blank(),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          axis.text.y=element_blank(),
          axis.ticks.y=element_blank(),
          legend.position = "right",
          legend.title = element_blank()) -> p
  ggsave(paste0("/Users/bassler/Dropbox/Hackathon/Move_plot/Plots_User_",n,"_move.png"), plot=p)
}

#####Eat#####
for (n in good_users){
  test_data <- full_data [full_data$ID %in% user_list[n] & full_data$final_activity == "eat",]
  ggplot(test_data, aes(x=as.Date(activity_day), y=..density..)) + 
    geom_density(aes(fill=recognized_activity),color=NA, position="stack")+
    scale_x_date(date_labels = '%b-%Y', date_breaks  ="3 month", limits = c(as.Date("2020-02-01"), as.Date("2021-03-01")))+
    theme(axis.text.x = element_text(angle=75, vjust=0.6, size=12, family="SF Pro Display"),
          panel.background = element_blank(),#, axis.line = element_line(colour = "black"),
          plot.title = element_blank(),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          axis.text.y=element_blank(),
          axis.ticks.y=element_blank(),
          legend.position = "right",
          legend.title = element_blank())+
    labs(title=paste0("User ",user, " profile"),
         #subtitle="City Mileage grouped by Class of vehicle",
         legend="Recognized activity") -> p
  ggsave(paste0("/Users/bassler/Dropbox/Hackathon/Eat_plot/Plots_User_",n,"_eat.png"), plot=p)
}


#####Recharge#####
for (n in good_users){
  test_data <- full_data [full_data$ID %in% user_list[n] & full_data$final_activity == "recharge",]
  ggplot(test_data, aes(x=as.Date(activity_day), y=..density..)) + 
    geom_density(aes(fill=recognized_activity),color=NA, position="stack")+
    scale_x_date(date_labels = '%b-%Y', date_breaks  ="3 month", limits = c(as.Date("2020-02-01"), as.Date("2021-03-01")))+
    theme(axis.text.x = element_text(angle=75, vjust=0.6, size=12, family="SF Pro Display"),
          panel.background = element_blank(),#, axis.line = element_line(colour = "black"),
          plot.title = element_blank(),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          axis.text.y=element_blank(),
          axis.ticks.y=element_blank(),
          legend.position = "right",
          legend.title = element_blank())+
    labs(title=paste0("User ",user, " profile"),
         #subtitle="City Mileage grouped by Class of vehicle",
         legend="Recognized activity") -> p
  ggsave(paste0("/Users/bassler/Dropbox/Hackathon/Recharge_plot/Plots_User_",n,"_recharge.png"), plot=p)
}


#####Summary#####
final_user <-user_list[4]
sle <- sleep_data[sleep_data$id == final_user,]
mov <- move_data[move_data$userId == final_user,]
test_data$ID <- final_user

test_data$date <- test_data$activity_day
final_data <- left_join(sle, mov, test_data, by=c("final_activity","date"))

final_act <- c(sle$final_activity, mov$final_activity, test_data$final_activity)
final_dat <- c(sle$date, mov$date, test_data$date)
final_other <- data.frame(final_act, final_dat)
colnames(final_other) <- c("final_activity", "date")





ggplot(final_other, aes(x = date, y = final_activity, fill=..y..))+
  scale_x_date(date_labels = '%d-%b', date_breaks  ="3 days")+#, limits = c(as.Date("2020-02-01"), as.Date("2021-03-01")))+
  theme(axis.text.x = element_text(angle=75, vjust=0.6, size=12, family="SF Pro Display"))+
  geom_density_ridges_gradient(scale = 5, rel_min_height = 0.0001, color = NA)+
  theme(axis.text.x = element_text(angle=75, vjust=0.6, size=14, family="SF Pro Display"),
        panel.background = element_blank(),#, axis.line = element_line(colour = "black"),
        plot.title = element_blank(),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        axis.text.y = element_text(size=20, family="SF Pro Display"),
        legend.position = "none",
        legend.title = element_blank())-> p
ggsave(paste0("/Users/bassler/Desktop/Documents/Hackathon_ZI/Summary_plot/FInal_plot_",n,"5.png"), plot=p)#, width=5, height=2.83, dpi=500, limitsize = FALSE)


