package edu.berkeley.nlp.assignments.assign3.student;

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
public class Model1SoftEmAligner extends Model1HardEmAligner {

   /**
    * Number of EM iterations to run.
    */
   protected static final int NUM_EM_ITERATIONS = 20;
   
   public Model1SoftEmAligner() {
      nullDistortionLikelihood = 0.28;
   }

   /* (non-Javadoc)
    * @see edu.berkeley.nlp.assignments.assign3.student.Model1HardEmAligner#train(java.lang.Iterable)
    */
   public void train(Iterable<SentencePair> trainingData) {
      
      System.out.println("(Soft EM Model 1) Initializing pair counters ...");
      initializePairCounters(trainingData);
      //initializePairCountersBaseline(trainingData);
      
      // alignmentProbability[i] = P( a_j = i | f[j], e[i] )
      double[] alignmentProbability = new double[MAX_SENTENCE_LEN];
      
      for (int emIterNum = 0; emIterNum < NUM_EM_ITERATIONS; emIterNum++) {
         
         System.out.println("EM iteration # " + emIterNum + " / "
               + NUM_EM_ITERATIONS + " ...");

         CounterMap<Integer, Integer> newPairCounters =
            new CounterMap<Integer, Integer>();
         
         Counter<Integer> englishProb = new Counter<Integer>();
         
         // E step. Align the sentence pairs using the existing translation
         // probability (pairCounters).
         // M step. Update the translation probability.
         for (SentencePair sentencePair : trainingData) {
            
            int numEnglishWords = sentencePair.englishWords.size();
            int numFrenchWords = sentencePair.frenchWords.size();
            
            // Generate the English word index. Use the last one as NULL
            // alignment.
            for (int ei = 0; ei < numEnglishWords; ei++) {
               String englishWord = sentencePair.englishWords.get(ei);
               int eIndex = englishWordIndexer.addAndGetIndex(englishWord);
               englishIndexBuffer[ei] = eIndex;
            }
            englishIndexBuffer[numEnglishWords] = -1;
            
            // For each French word, update the expectation of alignment count.
            for (int fi = 0; fi < numFrenchWords; fi++) {
               String fWord = sentencePair.frenchWords.get(fi);
               int fWordIndex = frenchWordIndexer.addAndGetIndex(fWord);
               double sumProbability = 0;
               
               // Set the alignment probability for all English words for this
               // French word.
               for (int ei = 0; ei < numEnglishWords; ei++) {
                  double probability = distortionProbability(numEnglishWords)
                        * pairCounters.getCount(
                              englishIndexBuffer[ei], fWordIndex);
                  
                  alignmentProbability[ei] = probability;
                  sumProbability += probability;
               }
               
               // Set the alignment probability for this French word and NULL.
               alignmentProbability[numEnglishWords] = nullDistortionLikelihood
                  * pairCounters.getCount(-1, fWordIndex);
               sumProbability += alignmentProbability[numEnglishWords];

               // Normalize the alignment probability and update the pair
               // counts.
               for (int ei = 0; ei <= numEnglishWords; ei++) {
                  newPairCounters.incrementCount(englishIndexBuffer[ei],
                        fWordIndex,
                        alignmentProbability[ei] / sumProbability);
                  englishProb.incrementCount(englishIndexBuffer[ei],
                        alignmentProbability[ei] / sumProbability);
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
