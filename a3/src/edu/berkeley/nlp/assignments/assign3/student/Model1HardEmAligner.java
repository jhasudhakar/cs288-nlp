package edu.berkeley.nlp.assignments.assign3.student;

import edu.berkeley.nlp.mt.Alignment;
import edu.berkeley.nlp.mt.SentencePair;
import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.CounterMap;

/**
 * IBM Model 1 Alignment using soft EM.
 * 
 * The caller must call train() to train the model before using it.
 * 
 * @author rxin
 */
public class Model1HardEmAligner extends AlignerBase {
   
   /**
    * Number of EM iterations to run.
    */
   protected static final int NUM_EM_ITERATIONS = 5;
   
   /**
    * The distortion likelihood for aligning a word to NULL.
    */
   protected double nullDistortionLikelihood = 0.2;
   
   protected int[] englishIndexBuffer = new int[MAX_SENTENCE_LEN];
   protected int[] frenchIndexBuffer = new int[MAX_SENTENCE_LEN];
   
   /* (non-Javadoc)
    * @see edu.berkeley.nlp.mt.WordAligner#alignSentencePair(edu.berkeley.nlp.mt.SentencePair)
    */
   @Override
   public Alignment alignSentencePair(SentencePair sentencePair) {
      int[] a = align(sentencePair);
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
   protected double distortionProbability(int numEnglishWords) {
      return (1.0 - nullDistortionLikelihood) / (numEnglishWords + 1);
   }
   
   /**
    * Aligns a sentence pair. This assumes pairCounters has been initialized.
    * 
    * @param sentencePair
    * @return
    */
   protected int[] align(SentencePair sentencePair) {
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
               * pairCounters.getCount(-1, fWordIndex);
         alignments[fi] = -1;
         //System.out.println(fi + ",-1: " + bestProbability);
         
         // Find the most likely alignment.
         for (int ei = 0; ei < numEnglishWords; ei++) {
            int eWordIndex = englishIndexBuffer[ei];
            double probability = distortionProbability(numEnglishWords)
                  * pairCounters.getCount(eWordIndex, fWordIndex);
            
            //System.out.println(fi + "," + ei + ": " + probability);
            
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
   protected void initializePairCounters(Iterable<SentencePair> trainingData) {
      
      pairCounters = new CounterMap<Integer, Integer>();
      
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
         }
         
         // NULL word.
         englishIndexBuffer[numEnglishWords] = -1;
         
         for (int i = 0; i < numFrenchWords; i++) {
            String f = pair.frenchWords.get(i);
            int fIndex = frenchWordIndexer.addAndGetIndex(f);
            frenchIndexBuffer[i] = fIndex;
         }
         
         // Increment the frequency counts for <e, f> pairs.
         for (int fi = 0; fi < numFrenchWords; fi++) {
            int f = frenchIndexBuffer[fi];
            pairCounters.setCount(-1, f, 1);
            //for (int ei = 0; ei <= numEnglishWords; ei++) {
            for (int ei = 0; ei < numEnglishWords; ei++) {
               int e = englishIndexBuffer[ei];
               pairCounters.incrementCount(e, f, 1);
            }
         }
      }
      
      // Normalize the counts.
      pairCounters.normalize();
   }
   
   /* (non-Javadoc)
    * @see edu.berkeley.nlp.assignments.assign3.student.AlignerBase#train(java.lang.Iterable)
    */
   public void train(Iterable<SentencePair> trainingData) {
      
      System.out.println("Initializing pair counters ...");
      initializePairCounters(trainingData);
      //initializePairCountersBaseline(trainingData);
      
      for (int emIterNum = 0; emIterNum < NUM_EM_ITERATIONS; emIterNum++) {
         
         System.out.println("EM iteration # " + emIterNum + " ...");
         
         CounterMap<Integer, Integer> newPairCounters =
            new CounterMap<Integer, Integer>();
         
         Counter<Integer> englishProb = new Counter<Integer>();
         
         // E step. Find alignment of the highest probability.
         // M step. Update the translation probability (pairCount).
         for (SentencePair sentencePair : trainingData) {
            
            int[] a = align(sentencePair);
            
            int numFrenchWords = sentencePair.frenchWords.size();
            for (int fi = 0; fi < numFrenchWords; fi++) {
               if (a[fi] == -1) {
                  newPairCounters.incrementCount(-1, frenchIndexBuffer[fi], 1);
                  englishProb.incrementCount(-1, 1);
               } else {
                  newPairCounters.incrementCount(englishIndexBuffer[a[fi]],
                        frenchIndexBuffer[fi], 1);
                  englishProb.incrementCount(englishIndexBuffer[a[fi]], 1);
               }
            }
         }
         
         // Normalize the counts.
         newPairCounters.normalize();
         
         // Switch newPairCounters and pairCounters ...
         pairCounters = newPairCounters;
      }
   }

}
