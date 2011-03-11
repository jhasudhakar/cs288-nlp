package edu.berkeley.nlp.assignments.assign3.student;

import edu.berkeley.nlp.mt.Alignment;
import edu.berkeley.nlp.mt.SentencePair;

/**
 * Heuristic-based word alignment.
 * 
 * max train = 1,000,000
 * Precision: 0.5230769230769231
 * Recall: 0.5564635958395245
 * AER: 0.46473141616928926
 * 
 * max train = 400,000
 * Precision: 0.4891737891737892
 * Recall: 0.5141158989598811
 * AER: 0.5017182130584192
 * 
 * max train = 100,000
 * Precision: 0.4346153846153846
 * Recall: 0.45641406636948983
 * AER: 0.5574244890576958
 *
 * max train = 10,000
 * Precision: 0.31153846153846154
 * Recall: 0.30039623576027735
 * AER: 0.6925302948091879
 *
 * max train = 1,000
 * Precision: 0.22108262108262108
 * Recall: 0.1607231302625062
 * AER: 0.8009585820220655
 * 
 * max train = 100
 * Precision: 0.20754985754985755
 * Recall: 0.13620604259534422
 * AER: 0.8185024416711882
 *
 * max train = 10
 * Precision: 0.24985754985754985
 * Recall: 0.20975730559683012
 * AER: 0.7647856755290288
 *
 * max train = 1
 * Precision: 0.24757834757834757
 * Recall: 0.2055473006438831
 * AER: 0.7677699403147042
 * 
 * max train = 0
 * Precision: 0.24757834757834757
 * Recall: 0.2055473006438831
 * AER: 0.7677699403147042
 *
 * @author rxin
 *
 */
public class HeuristicAligner extends AlignerBase {
   
   /*
    * maxTrain | # English | # French
    * 100      | 1902      | 2145
    * 1000     | 3663      | 4253
    * 10000    | 10363     | 12957
    * 100000   | 25894     | 33709
    */
   
   
   /* (non-Javadoc)
    * @see edu.berkeley.nlp.mt.WordAligner#alignSentencePair(edu.berkeley.nlp.mt.SentencePair)
    */
   @Override
   public Alignment alignSentencePair(SentencePair sentencePair) {

      Alignment alignment = new Alignment();
      int numFrenchWords = sentencePair.getFrenchWords().size();
      int numEnglishWords = sentencePair.getEnglishWords().size();

      for (int ei = 0; ei < numEnglishWords; ei++) {

         String englishWord = sentencePair.englishWords.get(ei);
         if (englishWordIndexer.contains(englishWord)) {
            // Only align if we have seen the English words before. Now find
            // the French word that generates the highest
            // c(e, f) / (c(e)*c(f)).
            int e = englishWordIndexer.addAndGetIndex(englishWord);
            int bestFrenchWordPos = -1;
            double bestFrenchWordValue = 0.0;
            for (int fi = 0; fi < numFrenchWords; fi++) {
               String frenchWord = sentencePair.frenchWords.get(fi);
               if (frenchWordIndexer.contains(frenchWord)) {
                  int f = frenchWordIndexer.addAndGetIndex(frenchWord);
                  double v = pairCounters.getCount(e, f)
                        / englishWordCounter.get(e)
                        / foreignWordCounter.get(f);
                  if (v > bestFrenchWordValue) {
                     bestFrenchWordValue = v;
                     bestFrenchWordPos = fi;
                  }
               }
            }
            if (bestFrenchWordPos != -1) {
               alignment.addAlignment(ei, bestFrenchWordPos, true);
            }
         }
      }
      
      return alignment;
   }
   
   /**
    * Train the alignment.
    * @param trainingData
    */
   public void train(Iterable<SentencePair> trainingData) {
      
      System.out.println("Training starts ...............");

      // The two buffers are used to cache the word indexes and avoid
      // an extra lookup step.
      int[] englishIndexBuffer = new int[MAX_SENTENCE_LEN];
      int[] frenchIndexBuffer = new int[MAX_SENTENCE_LEN];
      
      for (SentencePair pair : trainingData) {
         
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
               pairCounters.incrementCount(e, f, 1);
            }
         }
      }
      
      System.out.println("Training ends ............... !");
      
      // Report some stats.
      System.out.println("# English Words: " + englishWordIndexer.size());
      System.out.println("# French Words: " + frenchWordIndexer.size());
      System.out.println("total pair counts: " + pairCounters.totalCount());
      System.out.println("total pair size: " + pairCounters.totalSize());
   }
}
