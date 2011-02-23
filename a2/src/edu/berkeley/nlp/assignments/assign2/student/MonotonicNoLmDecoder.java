package edu.berkeley.nlp.assignments.assign2.student;

import java.util.LinkedList;
import java.util.List;

import edu.berkeley.nlp.langmodel.NgramLanguageModel;
import edu.berkeley.nlp.mt.decoder.Decoder;
import edu.berkeley.nlp.mt.decoder.DistortionModel;
import edu.berkeley.nlp.mt.phrasetable.PhraseTable;
import edu.berkeley.nlp.mt.phrasetable.PhraseTableForSentence;
import edu.berkeley.nlp.mt.phrasetable.ScoredPhrasePairForSentence;

/**
 * Monotonic beam-search decoder with no language model.
 * 
 * Decoding took 1.035s.
 * 
 * BLEU score on test data was BLEU(17.237).
 * 
 * Total model score on test data was -5256.377266729055.
 * 
 * 
 * @author rxin
 * 
 */
public class MonotonicNoLmDecoder implements Decoder {
   
   private PhraseTable tm;

   @SuppressWarnings("unused")
   private NgramLanguageModel lm;

   @SuppressWarnings("unused")
   private DistortionModel dm;

   
   public MonotonicNoLmDecoder(PhraseTable tm, NgramLanguageModel lm,
         DistortionModel dm) {
      super();
      this.tm = tm;
      this.lm = lm;
      this.dm = dm;
   }


   /* (non-Javadoc)
    * @see edu.berkeley.nlp.mt.decoder.Decoder#decode(java.util.List)
    */
   @Override
   public List<ScoredPhrasePairForSentence> decode(List<String> frenchSentence) {
      int length = frenchSentence.size();
      PhraseTableForSentence tmState = tm.initialize(frenchSentence);
      List<ScoredPhrasePairForSentence> ret = new LinkedList<ScoredPhrasePairForSentence>();

      double scores[] = new double[length];
      for (int i = 0; i < length; i++) scores[i] = Double.NEGATIVE_INFINITY;
      
      ScoredPhrasePairForSentence backtraces[] = new ScoredPhrasePairForSentence[length];

      for (int fpos = 0; fpos < length; fpos++) {

         int start = fpos - tmState.getMaxPhraseLength() + 1;
         if (start < 0)
            start = 0;

         for (; start <= fpos; start++) {
            
            List<ScoredPhrasePairForSentence> translations = tmState
                  .getScoreSortedTranslationsForSpan(start, fpos + 1);

            if (translations != null) {
               for (final ScoredPhrasePairForSentence translation : translations) {
                  double score = translation.score;
                  if (start > 0) score += scores[start - 1];
                  if (score > scores[fpos]) {
                     scores[fpos] = score;
                     backtraces[fpos] = translation;
                  }
               }
            }
         }
      }
      
      // Need to back trace.
      int index = length - 1;
      while (index > 0) {
         ScoredPhrasePairForSentence translation = backtraces[index];
         ret.add(0, translation);
         index -= translation.getForeignLength();
      }
      
      return ret;
   }

}
