package edu.berkeley.nlp.assignments.assign3.student;

import edu.berkeley.nlp.langmodel.EnglishWordIndexer;
import edu.berkeley.nlp.mt.Alignment;
import edu.berkeley.nlp.mt.SentencePair;
import edu.berkeley.nlp.mt.WordAligner;
import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.CounterMap;
import edu.berkeley.nlp.util.StringIndexer;

/**
 * IBM Model 1 Alignment.
 * 
 * @author rxin
 */
public class Model1Aligner implements WordAligner {
   
   /**
    * Max number of words for a sentence. This is used to initialize
    * frenchWordIndexer and englishWordIndexer buffers.
    */
   protected static final int MAX_SENTENCE_LEN = 1024;
   
   /**
    * Number of EM iterations to run.
    */
   protected static final int NUM_EM_ITERATIONS = 10;
   
   /**
    * The distortion likelihood for aligning a word to NULL.
    */
   protected static final double nullDistortionLikelihood = 0.05;
   
   /**
    * French word indexer that maps a French word (java string) to an integer.
    */
   StringIndexer frenchWordIndexer = new StringIndexer();
   
   
   /**
    * English word indexer that maps an English word (java string) to an
    * integer.
    */
   StringIndexer englishWordIndexer = EnglishWordIndexer.getIndexer();

   
   /**
    * Number of times a translation <French word, English word> pair occurs.
    * We can achieve better locality if French word goes first (since English
    * is usually the inner loop in this aligner).
    */
   CounterMap<Integer, Integer> pairCounters = new CounterMap<Integer, Integer>();
   
   private int[] englishIndexBuffer = new int[MAX_SENTENCE_LEN];
   private int[] frenchIndexBuffer = new int[MAX_SENTENCE_LEN];
   
   public Model1Aligner(Iterable<SentencePair> trainingData) {      
      train(trainingData);
   }
   
   /* (non-Javadoc)
    * @see edu.berkeley.nlp.mt.WordAligner#alignSentencePair(edu.berkeley.nlp.mt.SentencePair)
    */
   @Override
   public Alignment alignSentencePair(SentencePair sentencePair) {
      int[] a = align(sentencePair, false);
      Alignment alignment = new Alignment();
      for (int i = 0; i < a.length; i++) {
         if (a[i] != -1) {
            alignment.addAlignment(a[i], i, true);
         }
      }
      
      return alignment;
   }
   
   /**
    * Distortion likelihood for IBM Model 1.
    * 
    * @param numEnglishWords
    * @return distortion likelihood, 0.8 / (|E| + 1).
    */
   private double distortionProbability(int numEnglishWords) {
      return (1.0 - nullDistortionLikelihood) / (numEnglishWords + 1);
   }
   
   private int[] align(SentencePair sentencePair, boolean isFirstIteration) {
      int numEnglishWords = sentencePair.englishWords.size();
      int numFrenchWords = sentencePair.frenchWords.size();
      int[] alignments = new int[numFrenchWords];
      
      // Generate the English word index.
      for (int ei = 0; ei < numEnglishWords; ei++) {
         String englishWord = sentencePair.englishWords.get(ei);
         int eIndex = englishWordIndexer.addAndGetIndex(englishWord);
         englishIndexBuffer[ei] = eIndex;
      }
      
      // For each French word, find the most likely alignment.
      for (int fi = 0; fi < numFrenchWords; fi++) {
         // First align the word to null (-1).
         String fWord = sentencePair.frenchWords.get(fi);
         int fWordIndex = frenchWordIndexer.addAndGetIndex(fWord);
         frenchIndexBuffer[fi] = fWordIndex;

         double bestProbability = nullDistortionLikelihood
               * pairCounters.getCount(fWordIndex, -1);
         
         // Find the most likely alignment.
         for (int ei = 0; ei < numEnglishWords; ei++) {
            int eWordIndex = englishIndexBuffer[ei];
            double probability = distortionProbability(numEnglishWords)
                  * pairCounters.getCount(fWordIndex, eWordIndex);
            
            if (probability > bestProbability) {
               bestProbability = probability;
               alignments[fi] = ei;
            }
         }
      }
      
      return alignments;
   }
   
   /**
    * Generate the initial word pair counts (translation probability). This
    * function uses the baseline aligner to set the initial probability.
    * 
    * The reason we don't use a simple uniform distribution for initial
    * probability is because if we do that, most words will be aligned to the
    * first word in the sentence, which happen to be "the", and converge at
    * that local optimum.
    * 
    * @param trainingData
    */
   @SuppressWarnings("unused")
   private void initializePairCountersBaseline(
         Iterable<SentencePair> trainingData) {
      
      pairCounters = new CounterMap<Integer, Integer>();
      
      DynamicIntArray englishWordCounter = new DynamicIntArray(1000);
      DynamicIntArray frenchWordCounter = new DynamicIntArray(1000);
      
      int nullCount = 0;
      int sentenceCount = 0;
      
      // Counting.
      for (SentencePair pair : trainingData) {
         
         sentenceCount ++;
         
         int numFrenchWords = pair.frenchWords.size();
         int numEnglishWords = pair.englishWords.size();
         
         for (int fi = 0; fi < numFrenchWords; fi++) {
            int fIndex = frenchWordIndexer.addAndGetIndex(
                  pair.frenchWords.get(fi));
            
            frenchWordCounter.inc(fIndex, 1);
            
            if (fi < numEnglishWords) {
               int eIndex = englishWordIndexer.addAndGetIndex(
                     pair.englishWords.get(fi));
               englishWordCounter.inc(eIndex, 1);
               pairCounters.incrementCount(fIndex, eIndex, 1);
            } else {
               nullCount++;
               pairCounters.incrementCount(fIndex, -1, 1);
            }
         }
      }
      
      // Normalize the counts.
      for (Integer f : pairCounters.keySet()) {
         Counter<Integer> fCounter = pairCounters.getCounter(f);
         for (Integer e : fCounter.keySet()) {
            double count = pairCounters.getCount(f, e);
            if (e == -1) {
               pairCounters.setCount(f, e,
                   count / frenchWordCounter.get(f) / nullCount);
            } else {
               pairCounters.setCount(f, e, count / frenchWordCounter.get(f)
                     / englishWordCounter.get(e));
            }
         }
      }
   }
   
   /**
    * Generate the initial word pair counts (translation probability). This
    * function sets the initial translation probability to
    * c(e, f) / (c(e)*c(f)).
    * 
    * The reason we don't use a simple uniform distribution for initial
    * probability is because if we do that, most words will be aligned to the
    * first word in the sentence, which happen to be "the", and converge at
    * that local optimum.
    * 
    * This function is very similar to the HeuristicAligner, except we swap the
    * order of English and French in pairCount data structure.
    * 
    * @param trainingData
    */
   @SuppressWarnings("unused")
   private void initializePairCounters(Iterable<SentencePair> trainingData) {
      
      pairCounters = new CounterMap<Integer, Integer>();
      
      DynamicIntArray englishWordCounter = new DynamicIntArray(1000);
      DynamicIntArray foreignWordCounter = new DynamicIntArray(1000);
      
      int sentenceCount = 0;
      
      for (SentencePair pair : trainingData) {
         
         sentenceCount ++;
         if (sentenceCount % 10000 == 0) {
            System.out.println("  init sentence " + sentenceCount);
         }
         
         int numEnglishWords = pair.englishWords.size();
         int numFrenchWords = pair.frenchWords.size();
         
         for (int i = 0; i < numEnglishWords; i++) {
            String e = pair.englishWords.get(i);
            int eIndex = englishWordIndexer.addAndGetIndex(e);
            englishIndexBuffer[i] = eIndex;
            
            // Increase the English word count.
            englishWordCounter.inc(eIndex, 1);
         }
         
         for (int i = 0; i < numFrenchWords; i++) {
            String f = pair.frenchWords.get(i);
            int fIndex = frenchWordIndexer.addAndGetIndex(f);
            frenchIndexBuffer[i] = fIndex;
            
            // Increase the French word count.
            foreignWordCounter.inc(fIndex, 1);
         }
         
         // Increment the frequency counts for <e, f> pairs.
         for (int ei = 0; ei < numEnglishWords; ei++) {
            for (int fi = 0; fi < numFrenchWords; fi++) {
               int e = englishIndexBuffer[ei];
               int f = frenchIndexBuffer[fi];
               pairCounters.incrementCount(f, e, 1);
            }
         }
      }
      
      // Normalize the counts.
      for (Integer f : pairCounters.keySet()) {
         Counter<Integer> fCounter = pairCounters.getCounter(f);
         for (Integer e : fCounter.keySet()) {
            double count = pairCounters.getCount(f, e);
            pairCounters.setCount(f, e,
                count / foreignWordCounter.get(f) / englishWordCounter.get(e));
         }
         
         // Also set the count for NULL.
         pairCounters.setCount(f, -1, 1.0 / sentenceCount);
      }
   }
   
   /**
    * Train the alignment.
    * @param trainingData
    */
   private void train(Iterable<SentencePair> trainingData) {
      
      System.out.println("Initializing pair counters ...");
      initializePairCounters(trainingData);
      //initializePairCountersBaseline(trainingData);
      
      for (int emIterNum = 0; emIterNum < NUM_EM_ITERATIONS; emIterNum++) {
         
         System.out.println("EM iteration # " + emIterNum + " ...");
         
         CounterMap<Integer, Integer> newPairCounters =
            new CounterMap<Integer, Integer>();
         
         // E step. Find alignment of the highest probability.
         // M step. Update the translation probability (pairCount).
         for (SentencePair sentencePair : trainingData) {
            
            int[] a = align(sentencePair, emIterNum == 0);
            
            int numFrenchWords = sentencePair.frenchWords.size();
            for (int fi = 0; fi < numFrenchWords; fi++) {
               newPairCounters.incrementCount(frenchIndexBuffer[fi],
                     englishIndexBuffer[a[fi]], 1);
            }
         }
         
         // Switch newPairCounters and pairCounters ...
         pairCounters = newPairCounters;
      }
   }

}
