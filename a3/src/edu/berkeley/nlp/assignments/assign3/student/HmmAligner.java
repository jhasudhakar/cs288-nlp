package edu.berkeley.nlp.assignments.assign3.student;

import java.util.List;

import edu.berkeley.nlp.mt.Alignment;
import edu.berkeley.nlp.mt.SentencePair;
import edu.berkeley.nlp.util.CounterMap;

/**
 * HMM-based word aligner, as proposed by Vogel 1996. This is done using a soft
 * EM algorithm.
 * 
 * @author rxin
 */
/**
 * @author rxin
 *
 */
public class HmmAligner extends Model1HardEmAligner {
   
   /**
    * Number of EM iterations to run.
    */
   protected static final int NUM_EM_ITERATIONS = 20;
   
   /**
    * The distortion likelihood for aligning a word to NULL.
    */
   protected double nullDistortionLikelihood = 0.2;
   
   /* (non-Javadoc)
    * @see edu.berkeley.nlp.mt.WordAligner#alignSentencePair(edu.berkeley.nlp.mt.SentencePair)
    */
   @Override
   public Alignment alignSentencePair(SentencePair sentencePair) {
      Alignment alignment = new Alignment();
      double[][] gamma = trainAlignmentForSentence(
            sentencePair.frenchWords, sentencePair.englishWords);
      
      //print2dArray(gamma);
      
      int numEnglishWords = sentencePair.englishWords.size();
      int numFrenchWords = sentencePair.frenchWords.size();
      
      for (int fi = 0; fi < numFrenchWords; fi++) {
         // Find the highest gamma value.
         int bestEnglishPosition = numEnglishWords;
         double bestGamma = gamma[fi][numEnglishWords];
         for (int ei = 0; ei < numEnglishWords; ei++) {
            if (gamma[fi][ei] > bestGamma) {
               bestEnglishPosition = ei;
               bestGamma = gamma[fi][ei];
            }
         }
         
         // Add the most likely position if it is not NULL.
         if (bestEnglishPosition != numEnglishWords) {
            alignment.addAlignment(bestEnglishPosition, fi, true);
         }
      }
      
      return alignment;
   }

   protected double[][] trainAlignmentForSentence(List<String> frenchWords,
         List<String> englishWords) {
      
      int numEnglishWords = englishWords.size();
      int numFrenchWords = frenchWords.size();
      
      // The +1 is for NULL, i.e. we store NULL as the last English word.
      double[][] alpha = new double[numFrenchWords][numEnglishWords + 1];
      double[][] gamma = new double[numFrenchWords][numEnglishWords + 1];
      
      // Construct transition matrix.
      double[][] trans = new double[numEnglishWords + 1][numEnglishWords + 1];
      for (int i = 0; i < numEnglishWords; i++) {
         trans[i][numEnglishWords] = nullDistortionLikelihood;
         trans[numEnglishWords][i] = (1 - nullDistortionLikelihood)
               / numEnglishWords;
         double norm = 0;
         for (int j = 0; j < numEnglishWords; j++) {
            trans[i][j] = Math.exp(- 2 * Math.abs(j - i - 1.1));
            norm += trans[i][j];
         }
         for (int j = 0; j < numEnglishWords; j++) {
            trans[i][j] = trans[i][j] * (1 - nullDistortionLikelihood) / norm;
         }
      }

      // Get the first French word.
      String firstFWord = frenchWords.get(0);
      int firstFWordIdx = frenchWordIndexer.addAndGetIndex(firstFWord);
      frenchIndexBuffer[0] = firstFWordIdx;
      
      // Generate the English word index. Use the last one as NULL
      // alignment. Also set the initial alpha.
      double normAlphaZero = 0;
      for (int ei = 0; ei < numEnglishWords; ei++) {
         String englishWord = englishWords.get(ei);
         int eIndex = englishWordIndexer.addAndGetIndex(englishWord);
         englishIndexBuffer[ei] = eIndex;
         
         // Initialize alpha.
         alpha[0][ei] = pairCounters.getCount(eIndex, firstFWordIdx)
               * trans[ei][numEnglishWords];
         //alpha[0][ei] = trans[ei][numEnglishWords];
         normAlphaZero += alpha[0][ei];
      }
      
      // Initialize the NULL word (and the last alpha[0] value).
      englishIndexBuffer[numEnglishWords] = -1;
      alpha[0][numEnglishWords] = pairCounters.getCount(-1, firstFWordIdx)
            * trans[numEnglishWords][numEnglishWords];
      //alpha[0][numEnglishWords] = trans[numEnglishWords][numEnglishWords];
      normAlphaZero += alpha[0][numEnglishWords];
      if (normAlphaZero != 0) {
         for (int ei = 0; ei <= numEnglishWords; ei++) {
            alpha[0][ei] /= normAlphaZero;
         }
      }
      
      // Calculate alpha by going forward.
      for (int fi = 1; fi < numFrenchWords; fi++) {
         String fWord = frenchWords.get(fi);
         int fWordIndex = frenchWordIndexer.addAndGetIndex(fWord);
         frenchIndexBuffer[fi] = fWordIndex;
         double norm = 0;

         for (int ei = 0; ei <= numEnglishWords; ei++) {
            for (int j = 0; j <= numEnglishWords; j++) {
               alpha[fi][ei] += alpha[fi - 1][j] * trans[j][ei];
            }
            alpha[fi][ei] *= pairCounters.getCount(englishIndexBuffer[ei],
                  fWordIndex);
            norm += alpha[fi][ei];
         }
         
         // Normalize alpha.
         if (norm > 0) {
            for (int ei = 0; ei <= numEnglishWords; ei++) {
               alpha[fi][ei] /= norm;
            }
         }
      }
      
      // Initialize gamma[numFrenchWords - 1].
      System.arraycopy(alpha[numFrenchWords - 1], 0, gamma[numFrenchWords - 1],
            0, numEnglishWords);
      
      // Used to buffer the normalization factors.
      double[] normFactors = new double[numEnglishWords + 1];
      
      // Calculate gamma by going backward.
      for (int fi = numFrenchWords - 2; fi >= 0; fi--) {
         
         // Calculate the normalization factors.
         for (int j = 0; j <= numEnglishWords; j++) {
            for (int j1 = 0; j1 <= numEnglishWords; j1++) {
               normFactors[j] += alpha[fi][j1] * trans[j1][j];
            }
         }
         
         // Calculate gammas.
         for (int ei = 0; ei <= numEnglishWords; ei++) {
            for (int j = 0; j <= numEnglishWords; j++) {
               
               if (normFactors[j] == 0) {
                  gamma[fi][ei] = 0;
               } else {
                  gamma[fi][ei] += alpha[fi][ei] * trans[ei][j]
                        * gamma[fi + 1][j] / normFactors[j];
               }
            }
         }
      }

      return gamma;
   }
   
   /* (non-Javadoc)
    * @see edu.berkeley.nlp.assignments.assign3.student.AlignerBase#train(java.lang.Iterable)
    */
   public void train(Iterable<SentencePair> trainingData) {
      
      System.out.println("(HMM) Initializing pair counters ...");
      initializePairCounters(trainingData);
            
      for (int emIterNum = 0; emIterNum < NUM_EM_ITERATIONS; emIterNum++) {
         
         System.out.println("EM iteration # " + emIterNum + " ...");
         
         CounterMap<Integer, Integer> newPairCounters =
            new CounterMap<Integer, Integer>();
         
         int sentenceCount = 0;
         
         for (SentencePair pair : trainingData) {
            sentenceCount ++;
            double[][] gamma = trainAlignmentForSentence(
                  pair.frenchWords, pair.englishWords);
            
            int numEnglishWords = pair.englishWords.size();
            int numFrenchWords = pair.frenchWords.size();
            
            for (int fi = 0; fi < numFrenchWords; fi++) {
               for (int ei = 0; ei <= numEnglishWords; ei++) {
                  newPairCounters.incrementCount(englishIndexBuffer[ei],
                        frenchIndexBuffer[fi], gamma[fi][ei]);
               }
            }
         }
         
         // Normalize the counts.
         newPairCounters.normalize();
      
         // Switch newPairCounters and pairCounters ...
         pairCounters = newPairCounters;
      }
   }
   
   protected void print2dArray(double[][] data) {
      System.out.println("---------------- 2d -------------------");
      for (int i = 0; i < data.length; i++) {
         for (int j = 0; j < data[i].length; j++) {
            System.out.print(data[i][j] + "\t");
         }
         System.out.println();
      }
   }

}
