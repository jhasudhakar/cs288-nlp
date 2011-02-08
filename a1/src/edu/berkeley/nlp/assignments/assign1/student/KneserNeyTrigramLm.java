package edu.berkeley.nlp.assignments.assign1.student;

import java.util.ArrayList;
import java.util.List;

import edu.berkeley.nlp.langmodel.EnglishWordIndexer;
import edu.berkeley.nlp.langmodel.NgramLanguageModel;
import edu.berkeley.nlp.util.StringIndexer;

/**
 * An trigram language model with Kneser-Ney smoothing.
 * 
 * @author rxin
 */
public class KneserNeyTrigramLm implements NgramLanguageModel {

   /**
    * Discounting factor in Kneser-Ney.
    * 0.70 -> 24.493
    * 0.75 -> 24.502
    * 0.80 -> 24.520
    * 0.90 -> 24.533
    * 0.95 -> 24.493
    */
   public static final float discount = 0.9f;
   
   /**
    * Hash table load factor. The larger this is, the less memory
    * we need to store the language model. However, performance 
    * degrades linearly when the factor goes beyond 0.7.
    * 0.70 -> 297.570s, 997M
    * 0.75 -> 315.191s, 950M
    * 0.80 -> 311.193s, 905M
    * 0.85 -> 331.348s, 881M
    * 0.90 -> 354.125s, 861M
    * 0.95 -> 710.900s, 836M
    */
   public static final float loadFactor = 0.75f;
   
   /**
    * Initial capacities for the counters and hash maps. Note that this
    * parameter should not affect the language model performance. It only helps
    * speed up the model training.
    */
   public static final int initial_unigram_capacity = 495200;
   public static final int initial_bigram_capacity  = 8375000;
   public static final int initial_trigram_capacity = 42000000;
   // Basic stats:
   // ---495,172 unigrams - 19 bits
   // -8,374,230 bigrams  - 23 bits
   // 41,627,672 trigrams - 26 bits
   
   // Unigram Size: 495172
   // Bigram Size: 8374230
   // Trigram Size: 29973319

   
   /**
    * Smaller capacity for the sanity test.
    */
   public static final int initial_unigram_capacity_small = 50000;
   public static final int initial_bigram_capacity_small  = 50000;
   public static final int initial_trigram_capacity_small = 50000;
   
   /**
    * A small value for probability.
    * 1e-6: 24.502
    */
   public static final double very_small_value = 1e-6;

   /**
    * Unigram word indexer and counter.
    */
   StringIndexer wordIndexer;
   int unigramCounter[];

   /**
    * Bigram indexer and bigram counter.
    */
   BigramIndexer bigramIndexer;
   int bigramCounter[];

   /**
    * Trigram counter. There is no need for an indexer here. Note this is the
    * data structure that consumes the most amount of memory.
    */
   TrigramCounterInterface trigramCounter;
   int totalTrigram = 0;

   /**
    * The three N1+(...).
    */
   int n1plus_x_unigram_x[];
   int n1plus_bigram_x[];
   int n1plus_x_bigram[];
   
   /**
    * Basic stats.
    */
   int num_trigrams;
   int num_bigrams;
   int num_unigrams;
   
   double unseenBigramLogProb = 0;
   double unseenTrigramLogProb = 0;

   public KneserNeyTrigramLm(Iterable<List<String>> sentenceCollection,
         boolean approximate) {

      if (Runtime.getRuntime().maxMemory() > 500 * 1024 * 1024) {
         init(initial_unigram_capacity, initial_bigram_capacity,
               initial_trigram_capacity, approximate);
      } else {
         init(initial_unigram_capacity_small, initial_bigram_capacity_small,
               initial_trigram_capacity_small, approximate);
      }
      
      buildModel(sentenceCollection);
   }

   /**
    * Allocate memory for the structures. The structures are allocated here so I
    * can better measure the memory consumption.
    */
   private void init(int unigram_cap, int bigram_cap, int trigram_cap,
         boolean approximate) {
     Utils.reportMemoryUsage();
      wordIndexer = EnglishWordIndexer.getIndexer();
      Utils.reportMemoryUsage();
      unigramCounter = new int[unigram_cap];
      Utils.reportMemoryUsage();
      bigramIndexer = new BigramIndexer(bigram_cap, loadFactor);
      Utils.reportMemoryUsage();
      bigramCounter = new int[bigram_cap];
      Utils.reportMemoryUsage();
      
      if (!approximate) {
         trigramCounter = new TrigramCounter(trigram_cap, loadFactor);
      } else {
         trigramCounter = new TrigramCounterApproximate(trigram_cap, loadFactor);
      }
      Utils.reportMemoryUsage();
      
      n1plus_x_unigram_x = new int[unigram_cap];
      Utils.reportMemoryUsage();
      n1plus_bigram_x = new int[bigram_cap];
      Utils.reportMemoryUsage();
      n1plus_x_bigram = new int[bigram_cap];
      Utils.reportMemoryUsage();
   }

   /* (non-Javadoc)
    * @see edu.berkeley.nlp.langmodel.NgramLanguageModel#getOrder()
    */
   @Override
   public int getOrder() {
      return 3;
   }

   /* (non-Javadoc)
    * @see edu.berkeley.nlp.langmodel.NgramLanguageModel#getNgramLogProbability(int[], int, int)
    */
   @Override
   public double getNgramLogProbability(int[] ngram, int from, int to) {
      if (to - from == 3) {
         // Assertion: Trigram.
         double prob = 0.0;
         int word2 = ngram[from + 1];

         int word1word2 = bigramIndexer.get(ngram[from], word2);
         int word2word3 = bigramIndexer.get(word2, ngram[from + 2]);

         int word1word2count = word1word2 > 0 ? bigramCounter[word1word2] : 1;

         if (word1word2 > 0 && word2word3 > 0) {
            int trigramCount = trigramCounter.get(word1word2, ngram[from + 2]);
            if (trigramCount > 0) {
               prob = (trigramCount - discount) / word1word2count;
            } // else prob = 0;
         } // else prob = 0;
         
         double bigram_x = word1word2 > 0
               ? n1plus_bigram_x[word1word2]
               : very_small_value;
         double x_bigram = word2word3 > 0
               ? n1plus_x_bigram[word2word3]
               : very_small_value;

         int x_unigram_x = (word2 <= 0 || word2 >= n1plus_x_unigram_x.length)
               ? 1
               : n1plus_x_unigram_x[word2];

         prob += discount * bigram_x * x_bigram / x_unigram_x / word1word2count;

         return Math.log(prob);

      } else {
         // Assertion: Bigram, beginning of a sentence.
         int word1word2 = bigramIndexer.get(ngram[from], ngram[from + 1]);
         // if (word1word2 == 0) return unseenBigramLogProb;
         double x_bigram = (bigramCounter[word1word2] > 0)
               ? n1plus_x_bigram[word1word2]
               : very_small_value;
         int word1count = (ngram[from] < 0 || ngram[from] >= n1plus_x_unigram_x.length)
               ? 1
               : n1plus_x_unigram_x[ngram[from]];
         if (word1count <= 0) {
            // return unseenBigramLogProb;
            return Math.log(very_small_value);
         } else {
            if (x_bigram <= 0)
               x_bigram = very_small_value;
            return Math.log(x_bigram / word1count);
         }
      }
   }

   /* (non-Javadoc)
    * @see edu.berkeley.nlp.langmodel.NgramLanguageModel#getCount(int[])
    */
   @Override
   public long getCount(int[] ngram) {
      long count = 0;
      if (ngram.length == 3) {
         int word1word2 = bigramIndexer.get(ngram[0], ngram[1]);
         if (word1word2 > 0) count = trigramCounter.get(word1word2, ngram[2]);
      } else if (ngram.length == 2) {
         int word1word2 = bigramIndexer.get(ngram[0], ngram[1]);
         if (word1word2 > 0 && word1word2 <= num_bigrams) count = bigramCounter[word1word2];
      } else if (ngram.length == 1) {
         if (ngram[0] > 0 && ngram[0] < num_unigrams)
         count = unigramCounter[ngram[0]];
      }
      return count;
   }

   /**
    * Build the language model using the input sentenceCollection.
    * @param sentenceCollection
    */
   private void buildModel(Iterable<List<String>> sentenceCollection) {
      System.out.println("Building the language model . . .");

      // Loop over all sentences.
      int num_sentence = 0;
      for (List<String> sentence : sentenceCollection) {
         num_sentence++;
         if (num_sentence % 1000000 == 0) {
            System.out.println("On sentence " + num_sentence);
            //Utils.reportMemoryUsage();
            //reportStatus();
         }

         // Pad the sentence with START and STOP.
         List<String> stoppedSentence = new ArrayList<String>(sentence);
         stoppedSentence.add(0, NgramLanguageModel.START);
         stoppedSentence.add(NgramLanguageModel.STOP);

         // Get the index for the first two words.
         int word1 = wordIndexer.addAndGetIndex(stoppedSentence.get(0));
         int word2 = wordIndexer.addAndGetIndex(stoppedSentence.get(1));
         unigramCounter[word1]++;
         unigramCounter[word2]++;
         int word1word2 = bigramIndexer.addAndGetIndex(word1, word2);
         bigramCounter[word1word2]++;

         // Go over all words.
         int num_words = stoppedSentence.size();
         for (int i = 2; i < num_words; i++) {
            // Get the unigram index and update the count.
            int word3 = wordIndexer.addAndGetIndex(stoppedSentence.get(i));
            unigramCounter[word3]++;

            // Get the bigram index and bigram count.
            int word2word3 = bigramIndexer.addAndGetIndex(word2, word3);
            bigramCounter[word2word3]++;

            // Count the trigram. If it's a new trigram, incease the appropriate
            // n1plus values.
            if (trigramCounter.increaseCount(word1word2, word3) == 1) {
               // Assertion: This is a new trigram.
               n1plus_x_unigram_x[word2]++;
               n1plus_bigram_x[word1word2]++;
               n1plus_x_bigram[word2word3]++;
            }
            
            // Update the prev word indexes.
            word1 = word2;
            word2 = word3;
            word1word2 = word2word3;
         }
      }
      
      // Update stats.
      num_unigrams = wordIndexer.size();
      num_bigrams = bigramIndexer.size();
      num_trigrams = trigramCounter.size();
      
      // Calculate log prob for an unseen bigram.
      int totalBigramCount = 0;
      for (int i : bigramCounter) totalBigramCount += i;
      unseenBigramLogProb = Math.log(1.0 / (totalBigramCount + 1.0));
      
      // Calculate log prob for an unseen trigram.
      unseenTrigramLogProb = Math.log(1.0 / (trigramCounter.sum() + 1.0));
      
      System.out.println("unseen bigram prob = " + unseenBigramLogProb);
      System.out.println("unseen trigram prob = " + unseenTrigramLogProb);
      
      setToOneIfZero(n1plus_x_unigram_x);
      setToOneIfZero(n1plus_bigram_x);
      setToOneIfZero(n1plus_x_bigram);
      setToOneIfZero(bigramCounter);
      setToOneIfZero(unigramCounter);

      // Finish up.
      System.out.println("Done building language model.");
      reportStatus();
      Utils.reportMemoryUsage();
   }
   
   /**
    * Set all elements in the array that are zero to one.
    */
   protected void setToOneIfZero(int arr[]) {
      for (int i = 0; i < arr.length; i++) {
         if (arr[i] <= 0) {
            arr[i] = 1;
         }
      }
   }
   
   /**
    * A helper function to check the counts.
    */
   protected void reportCount(String word1, String word2, String word3) {
      int w1 = wordIndexer.addAndGetIndex(word1);
      int w2 = wordIndexer.addAndGetIndex(word2);
      int w3 = wordIndexer.addAndGetIndex(word3);
      int w1array[] = {w1}, w2array[] = {w2}, w3array[] = {w3};
      int w1w2[] = {w1, w2}, w2w3[] = {w2, w3};
      int w1w2w3[] = {w1, w2, w3};
      System.out.println(word1 + ": " + getCount(w1array));
      System.out.println(word2 + ": " + getCount(w2array));
      System.out.println(word3 + ": " + getCount(w3array));
      System.out.println(word1 + "," + word2 + ": " + getCount(w1w2));
      System.out.println(word2 + "," + word3 + ": " + getCount(w2w3));
      System.out.println(word1 + "," + word2 + "," + word3 + ": " + getCount(w1w2w3));
   }

   protected void reportStatus() {
      System.out.println("Unigram Size: " + wordIndexer.size());
      System.out.println("Bigram Size: " + bigramIndexer.size());
      System.out.println("Trigram Size: " + trigramCounter.size());
      //System.out.println("Total trigram: " + totalTrigram + ", " + trigramCounter.sum());
      //trigramCounter.reportTopTrigram();
      System.out.println();
   }

   protected void reportTopUnigram(int start) {
      int topUnigram = 0;
      int topUnigramCount = 0;
      for (int i = start; i < unigramCounter.length; i++) {
         if (i != 22 && unigramCounter[i] > topUnigramCount) {
            topUnigram = i;
            topUnigramCount = unigramCounter[i];
         }
      }
      System.out.println("Top unigram (index, word, count): " + topUnigram
            + ", " + wordIndexer.get(topUnigram) + ", " + topUnigramCount);
   }
}