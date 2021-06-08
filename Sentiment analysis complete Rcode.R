#Getting the libraries
library(caTools)
library(dplyr)
library(ggplot2)
library(lattice)
library(lexicon)
library(magrittr)
library(readr)
library(reshape2)
library(rmarkdown)
library(SnowballC)
library(stringr)
library(syuzhet)
library(tidytext)
library(tidyverse)
library(tm)
library(wordcloud)
library(corrplot)


#######################################################################################################################################################
start_script <- Sys.time()

#1st Part Exploratory Data and Text Analysis

#Reading the original dataset (515,738 records and 17 variables)
review_dataset<-read.csv(file="Hotel_Reviews.csv", header=TRUE, sep=",", stringsAsFactors = FALSE)

#Combining the positive and negative reviews
positive_reviews<-review_dataset %>%
  mutate(Review = Positive_Review) %>%
  mutate(Word_Count=Review_Total_Positive_Word_Counts) %>%
  mutate(Review_Type= "Positive") %>%
  filter(Word_Count>0)%>%
  select(Review,Word_Count,Review_Type)

negative_reviews<-review_dataset %>%
  mutate(Review = Negative_Review) %>%
  mutate(Word_Count= Review_Total_Negative_Word_Counts) %>%
  mutate(Review_Type= "Negative") %>%
  filter(Word_Count > 0) %>%
  select(Review,Word_Count,Review_Type)

#Creating a combined dataset of 867,640 records with 4 variables (Reviews, Word_count, Review_Type, Ispositive (binary))
all_reviews_combined<-rbind(data.frame(positive_reviews,stringsAsFactors = FALSE, isPositive=1),
                            data.frame(negative_reviews,stringsAsFactors = FALSE, isPositive=0))

#Checking if there are any missing values
which(!complete.cases(all_reviews_combined))  
which(!complete.cases(all_reviews_combined$Review)) 

#How many negative VS positive labelled reviews Negative: 387848   Positive: 479792 
table(all_reviews_combined$Review_Type)
prop.table(table(all_reviews_combined$Review_Type)) # 45% Negative and 55% Positive

#Plotting the Word Count per Review Type to highlight patterns, it seems that the longest review is negative
ggplot(all_reviews_combined,aes(Word_Count, fill=Review_Type)) + geom_histogram(binwidth = 20)
max(all_reviews_combined$Word_Count)


#Reducing the size of the dataset to 50000 records to make it easier to create a document term matrix
set.seed(1516)
random_selection<-sample(1:nrow(all_reviews_combined), 50000, replace= F)
reduced_dataset<-slice(all_reviews_combined, random_selection)
prop.table(table(reduced_dataset$Review_Type)) #Keeping an even distribution between positive and negative

#Corpus creation
reduced_dataset_corpus<-Corpus(VectorSource(reduced_dataset$Review))

#Cleaning steps

#Transform all letters into lowercase
reduced_dataset_corpus_cleaned<-tm_map(reduced_dataset_corpus, tolower)

#Removing numbers
reduced_dataset_corpus_cleaned<-tm_map(reduced_dataset_corpus_cleaned, removeNumbers)

#Removing punctuations
reduced_dataset_corpus_cleaned<-tm_map(reduced_dataset_corpus_cleaned, removePunctuation)

#Removing Stopwords
reduced_dataset_corpus_cleaned<-tm_map(reduced_dataset_corpus_cleaned, removeWords, stopwords("english"))

#Removing Stopwords "nothing"
reduced_dataset_corpus_cleaned<-tm_map(reduced_dataset_corpus_cleaned, removeWords,c("nothing", "NOTHING", "Nothing"))

#Removing whitespace
reduced_dataset_corpus_cleaned<-tm_map(reduced_dataset_corpus_cleaned, stripWhitespace)

#Steming words
reduced_dataset_corpus_cleaned<-tm_map(reduced_dataset_corpus_cleaned, stemDocument)

#Inspect the cleaned corpus
inspect(reduced_dataset_corpus_cleaned[1:8])

#Creating the document term matrix for tokenization of corpus

#DTM helps to tokenize corpus, so we can count in each row how many times a term occurs
reduced_reviews_dtm<-TermDocumentMatrix((reduced_dataset_corpus_cleaned))

#Creating a matrix for Term Document Matrix
reduced_reviews_matrix<-as.matrix(reduced_reviews_dtm) 

#Sort each term by frequency
reduced_reviews_frequency<-sort(rowSums(reduced_reviews_matrix), decreasing=TRUE)

#Creating a dataframe with the most frequent terms
reduced_reviews_dataframe<-data.frame(Word=names(reduced_reviews_frequency), Freq=reduced_reviews_frequency)

#Printing the 10 most frequent terms
top_frequency<- head(reduced_reviews_dataframe,20)                                                 
top_frequency
sum(reduced_reviews_dataframe$Freq)

par(mar=c(8,7,6.1,5))     #Expanding bottom margin
barplot(reduced_reviews_dataframe[1:10,]$Freq, las = 3, 
        names.arg = reduced_reviews_dataframe[1:10,]$Word,
        col ="lightblue", 
        main ="Top 10 most frequent words",
        ylab = "Word frequencies", cex.lab =1.5, cex.names = 1.5, cex.axis = 1.0, cex.main=2)

#Printing wordcloud with 500 frequent terms with a max of 100 words
set.seed(1517)
wordcloud(words=reduced_reviews_dataframe$Word, freq=reduced_reviews_dataframe$Freq, min.freq=500, 
          max.words=100,size=4,scale=c(8,0.8), random.order = FALSE, rot.per=0.20,colors=brewer.pal(12,"Paired"))

#Frequent terms & Associations

#Frequent terms
findFreqTerms(reduced_reviews_dtm,400)

set.seed(1518)
#Frequent terms associations
findAssocs(reduced_reviews_dtm, terms="staff", corlimit=0.2)
findAssocs(reduced_reviews_dtm, terms="room", corlimit=0.2)
findAssocs(reduced_reviews_dtm, terms="locat", corlimit=0.2)
findAssocs(reduced_reviews_dtm, terms="breakfast", corlimit=0.2)
findAssocs(reduced_reviews_dtm, terms="bed", corlimit=0.2)
findAssocs(reduced_reviews_dtm, terms="friend", corlimit=0.2)
findAssocs(reduced_reviews_dtm, terms="great", corlimit=0.2)
findAssocs(reduced_reviews_dtm, terms="help", corlimit=0.2)
findAssocs(reduced_reviews_dtm, terms="small", corlimit=0.2)
findAssocs(reduced_reviews_dtm, terms="book", corlimit=0.2)

#Plotting a correlation matrix with the above terms

#Creating a manual matrix for correlation matrix
y<-matrix(1:144, nrow=12, dimnames=list(c("staff","room","breakfast","locate", "great", "help", "friend", "buffet", "includ", "small", "book", "double"), 
                                   c("staff","room","breakfast","locate", "great", "help", "friend", "buffet", "includ", "small", "book", "double")))

y[2,2]<-1
y[3,3]<-1
y[4,4]<-1
y[5,5]<-1
y[6,6]<-1
y[7,7]<-1
y[8,8]<-1
y[9,9]<-1
y[10,10]<-1
y[11,11]<-1
y[12,12]<-1

y[3,8]<-0.22
y[8,3]<-0.22
y[3,9]<-0.21
y[9,3]<-0.21

y[2,11]<-0.22
y[11,2]<-0.22
y[2,12]<-0.21
y[12,2]<-0.21
y[2,10]<-0.27
y[10,2]<-0.27

y[1,6]<-0.46
y[6,1]<-0.46
y[1,7]<-0.46
y[7,1]<-0.46

y[5,4]<-0.25
y[4,5]<-0.25

y[y>1]<-0
y

corrplot(y)

#######################################################################################################################################################

#2nd part - Lexicon-based approach
#Using the same environment as EDTA

#Creating a subset by extracting all the reviews mention the term "Staff", we should obtain 178,884 observations and 4 variables  
staff_subset<-all_reviews_combined[grep("staff", all_reviews_combined$Review), ]    

#Saving the staff_subset as csv
write.csv(staff_subset, file= "staff_subset.csv")

head(staff_subset)

mean(staff_subset$Word_Count)   #Average word count per review on Staff
object.size(staff_subset)       #not sure if it's needed
glimpse(staff_subset)            #not sure if it's needed

table(staff_subset$Review_Type)      #How many negative VS positive labelled reviews on Staff Negative: 28803   Positive: 150081 
prop.table(table(staff_subset$Review_Type)) # 16% of negative reviews VS 84% of positive review

#Create a corpus
staff_corpus<-Corpus(VectorSource(staff_subset$Review))
staff_corpus[[1]][1]
staff_subset$isPositive[178000]

#Converting into lowercase
staff_corpus<-tm_map(staff_corpus,tolower)

#Removing punctuation
staff_corpus<-tm_map(staff_corpus,removePunctuation)

#Removing Stopwords
staff_corpus<-tm_map(staff_corpus,removeWords, c(stopwords("english")))    #Should we remove staff as stopword?

#Removing Whitespace
staff_corpus<-tm_map(staff_corpus,stripWhitespace)
writeLines(strwrap(staff_corpus[[24]]$content, 200))

#Stemming
staff_corpus<-tm_map(staff_corpus,stemDocument)
writeLines(strwrap(staff_corpus[[24]]$content, 200))

#Tokenize the corpus
tokenized_staff_corpus <- lapply(staff_corpus, scan_tokenizer)


Staff_df<-data.frame(text=sapply(tokenized_staff_corpus,paste,collapse=" "), stringsAsFactors = FALSE)
head(Staff_df)

Staff_token <- Staff_df %>% 
  mutate(Review_number = row_number())%>%    #line number for sentence grouping
  unnest_tokens(word,text)                  #tokenization - sentence to words
Staff_token

#Inspect the Afinn, Bing and NRC lexicons
get_sentiments("afinn")
get_sentiments("bing")
get_sentiments("nrc")

# Use the afinn sentiment to see the allocated score for each term
Staff_token %>%
  inner_join(get_sentiments("afinn")) %>%   #applying the lexicon to the Staff_token
  count(value, sort = TRUE)

# Use the bing sentiment set to see the different sentiment scores from the Staff_token
Staff_token %>%
  inner_join(get_sentiments("bing")) %>%
  filter(!is.na(sentiment)) %>% count(sentiment, sort = TRUE)

# Use the nrc sentiment set to see the different sentiment scores from the Staff_token
Staff_token %>%
  inner_join(get_sentiments("nrc")) %>%
  count(sentiment, sort = TRUE)

# View the most common words for the afinn sentiment
Staff_afinn <- Staff_token %>%
  inner_join(get_sentiments("afinn")) %>%
  count(word, value, sort = TRUE) %>%
  ungroup()
Staff_afinn

# View the most common words for the bing sentiment
Staff_bing <- Staff_token %>%
  inner_join(get_sentiments("bing")) %>%
  count(word, sentiment, sort = TRUE)%>%
  ungroup()
Staff_bing

# View the most common words for the nrc sentiment
Staff_nrc <- Staff_token %>%
  inner_join(get_sentiments("nrc")) %>%
  count(word, sentiment, sort = TRUE)%>%
  ungroup()
Staff_nrc

# Plot a graph of the top 10 words that contribute to each of the sentiment score from the afinn dictionary
Staff_afinn %>%
  group_by(value) %>%
  top_n(10) %>%
  ggplot(aes(reorder(word, n), n, fill = value)) +
  geom_bar(alpha = 0.8, stat = "identity", show.legend = FALSE) +
  facet_wrap(~value, scales = "free_y") +
  labs(title = "AFINN lexicon", y = "Top 10 words that contribute to the sentiment lexicon", x = NULL) +
  coord_flip()



# Plot a graph of the top 10 words that contribute to each of the sentiment score from the bing dictionary
Staff_bing %>%
  group_by(sentiment) %>%
  top_n(10) %>%
  ggplot(aes(reorder(word, n), n, fill = sentiment)) +
  geom_bar(alpha = 0.8, stat = "identity", show.legend = FALSE, size=6) +
  facet_wrap(~sentiment, scales = "free_y") +
  labs(title = "BING lexicon - Top 10 sentiments per polarity", y = "Frequency", cex.axis=15, x = NULL) +
  theme(axis.title.x=element_text(size=15), axis.text.x  = element_text(angle=90, vjust=0.5, size=15), 
        axis.text.y  = element_text(vjust=0.5, size=18), plot.title=element_text(size=20), strip.text = element_text(size=20))+
  coord_flip()


#Bing comparison-cloud
Staff_token %>%
  inner_join(get_sentiments("bing")) %>%
  count(word, sentiment, sort = TRUE) %>%
  acast(word ~ sentiment, value.var = "n", fill = 0) %>%
  comparison.cloud(colors = c("red2", "chartreuse4"),
                   max.words = 100, width=14, height=10)


# Plot a graph of the top 10 words that contribute to each of the sentiment score from the nrc dictionary
Staff_nrc %>%
  group_by(sentiment) %>%
  top_n(10) %>%
  ggplot(aes(reorder(word, n), n, fill = sentiment)) +
  geom_bar(alpha = 0.8, stat = "identity", show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free_y") +
  labs(title = "NRC lexicon", y = "Top 10 words that contribute to the sentimentlexicon", x = NULL) +
  coord_flip()


# Calculate and estimate the net sentiment for each tweet by each sentiment dictionary.

afinn <- Staff_token %>% 
  inner_join(get_sentiments("afinn")) %>% 
  group_by(Review_number) %>%
  summarise(sentiment = sum(value)) %>% 
  mutate(dictionary = "AFINN")

bing_and_nrc <- bind_rows(
  Staff_token %>% 
    inner_join(get_sentiments("bing")) %>%
    mutate(dictionary = "Bing et al."),
  Staff_token %>% 
    inner_join(get_sentiments("nrc") %>% 
                 filter(sentiment %in% c("positive", 
                                         "negative"))
    ) %>%
    mutate(dictionary = "NRC")) %>%
  count(dictionary, Review_number, sentiment) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment = positive - negative)

#
# Bind the results and generate a line graph for comparison (code can take at least 4 minutes to run)

start_time <- Sys.time()

bind_rows(afinn, 
          bing_and_nrc) %>%
  ggplot(aes(Review_number,sentiment, fill = dictionary)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~dictionary, ncol = 1, scales = "free_y")

end_time <- Sys.time()
runtime<-end_time-start_time
runtime
print(paste("It takes", runtime, "minutes, to run the lexicon comparison graph"))

#######################################################################################################################################################

#3rd part - Machine Learning - Polarity classification - Dataset pre-processing

#Checking the proportion of positive VS negative reviews

#How many negative VS positive labelled reviews on Staff Negative: 28803   Positive: 150081 
table(staff_subset$Review_Type)  

# 16% of negative reviews VS 84% of positive review
prop.table(table(staff_subset$Review_Type))
barplot(prop.table(table(staff_subset$Review_Type)),
        col=rainbow(3),
        ylim=c(0,1.0),
        main="Polarity distribution")

#Checking distribution of Negative observations
negative<-staff_subset[staff_subset$Review_Type=="Negative",]
nrow(negative)

#Checking distribution of Positive observations
positive<-staff_subset[staff_subset$Review_Type=="Positive",]
nrow(positive)

#Oversampling the Negative observation to a defined amount of records 25000 to make it equal
staff_subset_oversampling<-negative[sample(nrow(negative),25000),]
head(staff_subset_oversampling)
nrow(staff_subset_oversampling)

#Undersampling the Positive observation to a defined amount of records 25000 to make it equal
staff_subset_undersampling<-positive[sample(nrow(positive),25000),]
head(staff_subset_undersampling)
nrow(staff_subset_undersampling)


staff_subset_resampled<-rbind(data.frame(staff_subset_oversampling,stringsAsFactors = FALSE),
                              data.frame(staff_subset_undersampling,stringsAsFactors = FALSE))


#Checking the proportion of positive VS negative reviews of the resampled staff subset
table(staff_subset_resampled$Review_Type)      #How many negative VS positive labelled reviews on Staff Negative: 25000 and Positive: 25000
prop.table(table(staff_subset_resampled$Review_Type)) # 25% of negative reviews VS 25% of positive review
barplot(prop.table(table(staff_subset_resampled$Review_Type)),
        col=rainbow(3),
        ylim=c(0,1.0),
        main="Polarity distribution of the 're-sampled' staff dataset")


#Reducing the amount of records by selecting randomly 50 K records
set.seed(2469)
random_staff_records<-sample(1:nrow(staff_subset_resampled), 50000, replace= F)
smaller_staff_dataset<-slice(staff_subset_resampled, random_staff_records)

#Checking that the dataset is still evenly distributed
prop.table(table(smaller_staff_dataset$Review_Type))

#Creating the staff corpus
staff_corpus2<-Corpus(VectorSource(smaller_staff_dataset$Review))

#Cleaning of the staff corpus
#Transform all letters into lowercase
staff_corpus_clean<-tm_map(staff_corpus2, tolower)

#Removing numbers
staff_corpus_clean<-tm_map(staff_corpus_clean, removeNumbers)

#Removing punctuations
staff_corpus_clean<-tm_map(staff_corpus_clean, removePunctuation)

#Removing Stopwords (somes words won't bring insights because we know it's about staff, hotel, room,location)
staff_corpus_clean<-tm_map(staff_corpus_clean, removeWords, c("nothing", "staff", "hotel", "room", "rooms","locat","also",stopwords("english")))
staff_corpus_clean<-tm_map(staff_corpus_clean, removeWords, c("locat","location"))

#Removing whitespace
staff_corpus_clean<-tm_map(staff_corpus_clean, stripWhitespace)

#Steming words
staff_corpus_clean<-tm_map(staff_corpus_clean, stemDocument)

#Inspect the cleaned corpus
inspect(staff_corpus_clean[1:3])
staff_corpus_clean[[3]]

#Creating the document term matrix for tokenization of corpus and using TF-IDF technique 
staff_reviews_dtm<-DocumentTermMatrix(staff_corpus_clean, control=list(weighting=weightTfIdf))
staff_reviews_dtm

#Removing sparse terms to remove the terms that appear less than 0.1%
staff_reviews_dtm<-removeSparseTerms(staff_reviews_dtm, 0.99)
staff_reviews_dtm
inspect(staff_reviews_dtm[1,1:122])
inspect(staff_reviews_dtm[1:10,1:122])
staff_reviews_dtm

#Frequent terms
findFreqTerms(staff_reviews_dtm,500)
frequent_words<-data.frame(sort(colSums(as.matrix(staff_reviews_dtm)), decreasing=TRUE))
head(frequent_words,25)

#Finalizing the cleaned staff dataset by combining the document term frequency with the smaller dataset itself
smaller_staff_dataset=cbind(smaller_staff_dataset,as.matrix(staff_reviews_dtm))
smaller_staff_dataset$Review_Type=as.factor(smaller_staff_dataset$Review_Type)

#Removing columns that won't be used when running the algorithms in the second part using Python
smaller_staff_cleaned_dataset<-smaller_staff_dataset[,!(names(smaller_staff_dataset) %in% c("Review"))]   

#Savind the CSV cleaned and preprocessed staff dataset to run the algorithms with  Python

#Saving in R environment
write.csv(smaller_staff_cleaned_dataset, file="smaller_staff_cleaned_dataset.csv", row.names=FALSE)

#Saving in Python environment
write.csv(smaller_staff_cleaned_dataset, "C://Users/magno/smaller_staff_cleaned_dataset.csv", row.names=FALSE)

end_script <- Sys.time()

script_time<-end_script-start_script

print(paste("It takes", script_time, "minutes, to run the whole R script"))

