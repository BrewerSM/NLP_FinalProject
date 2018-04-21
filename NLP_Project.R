###NLP Final Project : UNH Master's in Analytics and Data Science###

#by:Sarah Brewer
#brewer.sm@gmail.com

##############################################################################
# This project utilizes several text mining techniques on headlines from US
# news sources. Headlines are primarily for economic/financial news. The size
# of the dataset is ~8,000 headlines from 1951-2014. Data was downloaded from
# https://www.figure-eight.com/data-for-everyone/. 
##############################################################################

#Import the required libraries and read in the headlines CSV.
library(qdap)
library(tm)
library(wordcloud)
library(RWeka)
library(plyr)
library(stringr)

mytext <- read.csv("C:/Users/Sarah/Desktop/Full-Economic-News-DFE-839861.csv", header = TRUE, sep = ",")
mytext$headline <- iconv(mytext$headline, from = "UTF-8", to = "ASCII", sub = "")

#Make the vector source (column of the CSV with the headlines) into a corpus.
news_corpus <- VCorpus(VectorSource(mytext$headline))

#Create a function to clean the corpus - replace abbreviations, make everything lower case, 
#remove punctuation,strip white space, etc.
clean_corpus <- function(cleaned_corpus){
  cleaned_corpus <- tm_map(cleaned_corpus, content_transformer(replace_abbreviation))
  cleaned_corpus <- tm_map(cleaned_corpus, content_transformer(tolower))
  cleaned_corpus <- tm_map(cleaned_corpus, removePunctuation)
  cleaned_corpus <- tm_map(cleaned_corpus, removeNumbers)
  cleaned_corpus <- tm_map(cleaned_corpus, removeWords, stopwords("english"))
  cleaned_corpus <- tm_map(cleaned_corpus, stripWhitespace)
  return(cleaned_corpus)
}

#Apply the function to the corpus we created.
cleaned_news_corpus <- clean_corpus(news_corpus)

#The following lines compare the same headline before it was cleaned and after.
print(news_corpus[[16]][1])
print(cleaned_news_corpus[[16]][1])


#Create a term document matrix out of the cleaned news corpus. Every column represents
#a news headline, while the rows represent words appearing in the news corpus.
TDM_news <- TermDocumentMatrix(cleaned_news_corpus)
TDM_news_m <- as.matrix(TDM_news)

#The following line views some of that TDM for the last few headlines.
TDM_news_m[1000:1010, 7990:8000]

#To get term frequency, create sums of the counts of word appearance across the corpus.
term_frequency <- rowSums(TDM_news_m)

#Sort term_frequency in descending order to view some of the most common words.
term_frequency <- sort(term_frequency,dec=TRUE)
top20 <- term_frequency[1:20]
top20

#Plot a barchart of the 20 most common words. Number 1 is "stock", number 2 is "fed".
barplot(top10,col="blue",las=2)

#The term frequencies can be used to build a word cloud. First, the term frequencies are
#converted to a data frame.
word_freqs <- data.frame(term = names(term_frequency), num = term_frequency)
wordcloud(word_freqs$term, word_freqs$num,max.words=100,colors=brewer.pal(8, "Paired"))

#The previous word cloud was a unigram word cloud. A bigram can also be built by tokenizing
#terms and creating a TDM with frequencies of bigrams. First,a function is used to tokenize.
tokenizer <- function(x)
  NGramTokenizer(x,Weka_control(min=2,max=2))

#This function is then applied to the cleaned news corpus to get the bigram TDM.
bigram_tdm <- TermDocumentMatrix(cleaned_news_corpus,control = list(tokenize=tokenizer))
bigram_tdm_m <- as.matrix(bigram_tdm)

#As before, frequencies are created as sums of counts, then sorted. This time the top
#words are "interest rates" and "wall street."
term_frequency <- rowSums(bigram_tdm_m)
term_frequency <- sort(term_frequency,dec=TRUE)

#The frequencies are put into a data frame, and the word cloud can be built.
word_freqs <- data.frame(term = names(term_frequency), num = term_frequency)
wordcloud(word_freqs$term, word_freqs$num,min.freq=5,max.words=100,colors=brewer.pal(8, "Paired"))

#A trigram, or 3-word pairs, can be created in the same way as the bigram. The only
#change is to adjust the arguments "min" and "max" to 3 in the tokenizer function.
tokenizer <- function(x)
  NGramTokenizer(x,Weka_control(min=3,max=3))

trigram_tdm <- TermDocumentMatrix(cleaned_news_corpus,control = list(tokenize=tokenizer))
trigram_tdm_m <- as.matrix(trigram_tdm)

#Now the top words are "wall street journal" and "review outlook editorial."
term_frequency <- rowSums(trigram_tdm_m)
term_frequency <- sort(term_frequency,dec=TRUE)

word_freqs <- data.frame(term = names(term_frequency), num = term_frequency)
wordcloud(word_freqs$term, word_freqs$num,min.freq=5,max.words=100,colors=brewer.pal(8, "Paired"))

#A different type of word cloud is TF-IDF based. This is created like the other word clouds, but
#using TF-IDF weighting.
tfidf_tdm <- TermDocumentMatrix(cleaned_news_corpus,control=list(weighting=weightTfIdf))
tfidf_tdm_m <- as.matrix(tfidf_tdm)

#This time, the top term is "digest" followed by "business" and "finance."
term_frequency <- rowSums(tfidf_tdm_m)
term_frequency <- sort(term_frequency,dec=TRUE)

word_freqs <- data.frame(term = names(term_frequency), num = term_frequency)

wordcloud(word_freqs$term, word_freqs$num,min.freq=5,max.words=100,colors=brewer.pal(8, "Paired"))

#The sentiment (positive or negative) of a headline can be assessed using polarity.
#The polarity score can then be used to divide the corpus into two in order to make
#comparison/commonality word clouds.

#First a function is written to get the polarity of a headline.
functionPol <- function(x){
  Pol <- polarity(x)
  Result <- Pol$all$polarity
  return (Result)
}

#This function is then applied to the headlines, with the polarity score added back to
#the original data.
Pol <- lapply(mytext$headline, functionPol)
mytext$Polarity <- sapply(Pol, paste0)

#The column "Polarity" has the type character, so it is converted to numeric.
mytext$Polarity <- as.numeric(mytext$Polarity)

#Now the data can be subset by positive and negative headlines.
pos <- mytext[mytext$Polarity>0,]
neg <- mytext[mytext$Polarity<0,]

#The text from the positive and negatives headlines can be extracted as a vector,
#then combined into a list of length 2.
posText <- as.vector(pos$headline)
negText <- as.vector(neg$headline)
PosNeg <- list(posText, negText)

#The list is then converted to a corpus and cleaned using the earlier cleaning function.
PosNegCorpus <- VCorpus(VectorSource(PosNeg))
cleaned_PosNeg_corpus <- clean_corpus(PosNegCorpus)

#As before, a TDM is generated for the corpus and converted to a matrix to be used 
#for the clouds.
TDM_PosNeg <- TermDocumentMatrix(cleaned_PosNeg_corpus)
colnames(TDM_PosNeg) <- c("Positive","Negative")
TDM_PosNeg_m <- as.matrix(TDM_PosNeg)

#A commonality cloud plots words shared across the positive and negative headlines.
commonality.cloud(TDM_PosNeg_m,colors=brewer.pal(8, "Dark2"),max.words = 100)

#The comparison cloud compares the frequencies of words in the positive and negative headlines.
#Positive headlines have words like "gains" and "strong," whereas negative headlines
#more frequently have words like "fall" and "jobless."
comparison.cloud(TDM_PosNeg_m,colors=brewer.pal(8, "Dark2"),max.words = 100)


#In addition to word clouds, an emotional radar chart can be built using the NRC lexicon.
#First tidy has to be used on the original TDM of the headlines. This stacks each headline's 
#index number vertically, along with the words in that headline and their count.
mytext_tidy <- tidy(TDM_news)

#Next sentiments are loaded from the NRC package. This is a corpus of words and their
#associated sentiment.
nrc_lex <- get_sentiments("nrc")

#The corpus of sentiments is joined with the tidy TDM on the common words in each.
story_nrc <- inner_join(mytext_tidy, nrc_lex, by = c("term" = "word"))

#In the following line, rows with the sentiment "positive" or "negative" are dropped, 
#as these dominate the sentiments. 
story_nrc_noposneg <- story_nrc[!(story_nrc$sentiment %in% c("positive","negative")),]

#The data frame is then aggregated by sentiment with a sum of counts in order to find prevailing 
#sentiments. This step can also be performed on "story_nrc" without positive and negative dropped to 
#see how they make up a majority.
aggdata <- aggregate(story_nrc_noposneg$count, list(index = story_nrc_noposneg$sentiment), sum)

#Finally, the emotional radar chart can be created.
chartJSRadar(aggdata)
