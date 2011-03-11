package edu.berkeley.nlp.assignments.assign3.student;

import edu.berkeley.nlp.mt.Alignment;
import edu.berkeley.nlp.mt.SentencePair;
import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.CounterMap;

/**
 * HMM-based word aligner, as proposed by Vogel 1996. This is done using a soft
 * EM algorithm.
 * 
 * @author rxin
 */
public class HmmAligner extends Model1HardEmAligner {
   
   /**
    * Number of EM iterations to run.
    */
   protected static final int NUM_EM_ITERATIONS = 20;
   
   /**
    * The distortion likelihood for aligning a word to NULL.
    */
   protected double nullDistortionLikelihood = 0.25;
   
   /* (non-Javadoc)
    * @see edu.berkeley.nlp.mt.WordAligner#alignSentencePair(edu.berkeley.nlp.mt.SentencePair)
    */
   @Override
   public Alignment alignSentencePair(SentencePair sentencePair) {
      // TODO Auto-generated method stub
      return null;
   }
   
   /* (non-Javadoc)
    * @see edu.berkeley.nlp.assignments.assign3.student.AlignerBase#train(java.lang.Iterable)
    */
   public void train(Iterable<SentencePair> trainingData) {
      
      System.out.println("(HMM) Initializing pair counters ...");
      initializePairCounters(trainingData);
      
      double[] initialState = new double[MAX_SENTENCE_LEN];
      
      for (int emIterNum = 0; emIterNum < NUM_EM_ITERATIONS; emIterNum++) {
         
         System.out.println("EM iteration # " + emIterNum + " ...");
         
         CounterMap<Integer, Integer> newPairCounters =
            new CounterMap<Integer, Integer>();
         
         Counter<Integer> englishProb = new Counter<Integer>();
         
         for (SentencePair sentencePair : trainingData) {
            
            int numEnglishWords = sentencePair.englishWords.size();
            int numFrenchWords = sentencePair.frenchWords.size();
            
            // Get the first French word.
            String firstFWord = sentencePair.frenchWords.get(0);
            int firstFWordIdx = frenchWordIndexer.addAndGetIndex(firstFWord);
            
            // Generate the English word index. Use the last one as NULL
            // alignment. Also set the initial state of the HMM.
            for (int ei = 0; ei < numEnglishWords; ei++) {
               String englishWord = sentencePair.englishWords.get(ei);
               int eIndex = englishWordIndexer.addAndGetIndex(englishWord);
               englishIndexBuffer[ei] = eIndex;
               initialState[ei] = pairCounters.getCount(firstFWordIdx, eIndex);
            }
            englishIndexBuffer[numEnglishWords] = -1;
            
         }
      
         // Switch newPairCounters and pairCounters ...
         pairCounters = newPairCounters;
      }
   }

}
