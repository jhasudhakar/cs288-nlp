package edu.berkeley.nlp.assignments.assign2.student;

import java.util.ArrayList;
import java.util.List;

import edu.berkeley.nlp.langmodel.EnglishWordIndexer;
import edu.berkeley.nlp.langmodel.NgramLanguageModel;
import edu.berkeley.nlp.mt.decoder.Decoder;
import edu.berkeley.nlp.mt.decoder.DistortionModel;
import edu.berkeley.nlp.mt.phrasetable.PhraseTable;
import edu.berkeley.nlp.mt.phrasetable.PhraseTableForSentence;
import edu.berkeley.nlp.mt.phrasetable.ScoredPhrasePairForSentence;
import edu.berkeley.nlp.util.FastPriorityQueue;

/**
 * A beam search which permits limited distortion.
 * 
 * @author rxin
 * 
 */
public class DistortingWithLmDecoder implements Decoder {
  
  public static final int priorityQueueSize = 100;

  public class BeamSearchOption {
    public int[] lmContextBuf;
    double score;
    int lmContextBufLen;
    
    // used for backtrace.
    ArrayList<ScoredPhrasePairForSentence> phrasePairs;
  }

  private PhraseTable tm;

  private NgramLanguageModel lm;

  private int lmOrder;
  
  private DistortionModel dm;

  public DistortingWithLmDecoder(PhraseTable tm, NgramLanguageModel lm,
      DistortionModel dm) {
    super();
    this.tm = tm;
    this.lm = lm;
    this.dm = dm;
    this.lmOrder = lm.getOrder();
  }

  /*
   * (non-Javadoc)
   * 
   * @see edu.berkeley.nlp.mt.decoder.Decoder#decode(java.util.List)
   */
  @SuppressWarnings("unchecked")
  @Override
  public List<ScoredPhrasePairForSentence> decode(List<String> frenchSentence) {
    int length = frenchSentence.size();
    PhraseTableForSentence tmState = tm.initialize(frenchSentence);

    int maxPhraseLen = tmState.getMaxPhraseLength();

    // Initialize the priority queues.
    FastPriorityQueue<BeamSearchOption> beams[] = new FastPriorityQueue[length + 1];
    for (int start = 0; start <= length; start++) {
      beams[start] = new FastPriorityQueue<BeamSearchOption>();
    }

    // Add search beam's root node.
    BeamSearchOption optionStart = new BeamSearchOption();
    optionStart.lmContextBufLen = 1;
    optionStart.lmContextBuf = new int[1];
    optionStart.lmContextBuf[0] = EnglishWordIndexer.getIndexer()
        .addAndGetIndex(NgramLanguageModel.START);
    optionStart.score = 0;
    beams[0].setPriority(optionStart, 0);

    for (int start = 0; start < length; start++) {
      
      // Copy the list of options at position start.
      //System.out.println("start postion #" + start + ": " + beams[start].size());
      List<BeamSearchOption> options = new ArrayList<BeamSearchOption>(beams[start].size());
      int count = 0;
      while (beams[start].hasNext() && count < priorityQueueSize) {
        options.add(beams[start].next());
        count ++;
      }
      beams[start] = null;

      for (int end = start + 1; end <= start + maxPhraseLen && end <= length; ++end) {

        List<ScoredPhrasePairForSentence> translations = tmState
            .getScoreSortedTranslationsForSpan(start, end);

        if (translations != null) {
          for (final ScoredPhrasePairForSentence translation : translations) {
            
            int[] newContextBuf = null;
            
            // If the translation phrase is long enough, only keep that as context.
            if (translation.english.indexedEnglish.length >= lmOrder - 1) {
              newContextBuf = new int[lmOrder - 1];
              System.arraycopy(translation.english.indexedEnglish,
                  translation.english.indexedEnglish.length - lmOrder + 1,
                  newContextBuf, 0,
                  lmOrder - 1);
            }

            for (BeamSearchOption option : options) {

              // Distortion ...
              int prevPhrases = 0;
              if (option.phrasePairs != null) {
                prevPhrases = option.phrasePairs.size();
              }
              
              int dIndex = 0;
              int d = 0;
              do {

                BeamSearchOption newOption = new BeamSearchOption();
                if (option.phrasePairs != null) {
                  newOption.phrasePairs = (ArrayList<ScoredPhrasePairForSentence>) option.phrasePairs
                      .clone();
                } else {
                  newOption.phrasePairs = new ArrayList<ScoredPhrasePairForSentence>(
                      1);
                }
                newOption.phrasePairs.add(prevPhrases - dIndex, translation);

                // Compute language model score.
                // Note this scoring mechanism is inefficient. We should be able
                // to reuse the previously calculated score to further compute
                // the score for this translation. But given it is already 2AM,
                // and I don't see how improving this efficiency will make me a
                // better computer scientist, I am leaving it as is.
                newOption.score = Decoder.StaticMethods.scoreHypothesis(
                    newOption.phrasePairs, lm, dm);
                beams[end].setPriority(newOption, newOption.score);

                // Next distortion position.
                dIndex++;
                if (dIndex < prevPhrases) {
                  d += option.phrasePairs.get(prevPhrases - dIndex).english.indexedEnglish.length;
                } else {
                  break;
                }
              } while (d <= dm.getDistortionLimit());
            }
          }
        }
      }
    }


    // Need to back trace.
    BeamSearchOption option = beams[length].getFirst();

    return option.phrasePairs;
  }

}
