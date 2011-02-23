package edu.berkeley.nlp.assignments.assign2.student;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import edu.berkeley.nlp.langmodel.EnglishWordIndexer;
import edu.berkeley.nlp.langmodel.NgramLanguageModel;
import edu.berkeley.nlp.mt.decoder.Decoder;
import edu.berkeley.nlp.mt.decoder.DistortionModel;
import edu.berkeley.nlp.mt.phrasetable.PhraseTable;
import edu.berkeley.nlp.mt.phrasetable.PhraseTableForSentence;
import edu.berkeley.nlp.mt.phrasetable.ScoredPhrasePairForSentence;
import edu.berkeley.nlp.util.FastPriorityQueue;
import edu.berkeley.nlp.util.GeneralPriorityQueue;
import edu.berkeley.nlp.util.PriorityQueue;

/**
 * Monotonic beam-search decoder with trigram language model.
 * 
 * @author rxin
 * 
 */
public class MonotonicWithLmDecoder implements Decoder {
  
  public static final int priorityQueueSize = 2000;

  public class BeamSearchOption {
    public int[] lmContextBuf;
    double score;
    int lmContextBufLen;
    
    // used for backtrace.
    ScoredPhrasePairForSentence phrasePair;
    BeamSearchOption prev;
  }

  private PhraseTable tm;

  private NgramLanguageModel lm;

  private int lmOrder;

  @SuppressWarnings("unused")
  private DistortionModel dm;

  public MonotonicWithLmDecoder(PhraseTable tm, NgramLanguageModel lm,
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
  @Override
  public List<ScoredPhrasePairForSentence> decode(List<String> frenchSentence) {
    int length = frenchSentence.size();
    PhraseTableForSentence tmState = tm.initialize(frenchSentence);
    List<ScoredPhrasePairForSentence> ret = new LinkedList<ScoredPhrasePairForSentence>();

    int maxPhraseLen = tmState.getMaxPhraseLength();
    int lmContextBufSize = lm.getOrder() + tmState.getMaxPhraseLength() + 1;
    int[] lmContextBuf = new int[lmContextBufSize];
    int currentContextBufLen = 0;

    // Initialize the priority queues.
    @SuppressWarnings("unchecked")
    GeneralPriorityQueue<BeamSearchOption> beams[] = new GeneralPriorityQueue[length + 1];
    for (int start = 0; start <= length; start++) {
      beams[start] = new GeneralPriorityQueue<BeamSearchOption>();
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

        //System.out.println("(start, end) = (" + start + ", " + end + ")");

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
              BeamSearchOption newOption = new BeamSearchOption();
              newOption.prev = option;
              newOption.phrasePair = translation;

              // Get the translation score.
              double score = option.score;
              score += translation.score;

              // Update the lm context for language model scoring.
              System.arraycopy(option.lmContextBuf, 0, lmContextBuf, 0,
                  option.lmContextBuf.length);
              System.arraycopy(translation.english.indexedEnglish, 0,
                  lmContextBuf, option.lmContextBuf.length,
                  translation.english.indexedEnglish.length);
              currentContextBufLen = option.lmContextBuf.length
                  + translation.english.indexedEnglish.length;

              if (end == length) {
                lmContextBuf[currentContextBufLen] = EnglishWordIndexer
                    .getIndexer().addAndGetIndex(NgramLanguageModel.STOP);
                currentContextBufLen++;
              }
              
              // Score ...
              score += scoreLm(option.lmContextBuf.length, lmContextBuf,
                  currentContextBufLen, lm);

              // Save the option into the priority queue.
              if (newContextBuf != null) {
                newOption.lmContextBuf = newContextBuf;
                newOption.lmContextBufLen = lmOrder - 1;
              } else {
                newOption.lmContextBufLen = option.lmContextBufLen + translation.english.indexedEnglish.length;
                if (newOption.lmContextBufLen > lmOrder - 1) {
                  newOption.lmContextBufLen = lmOrder - 1;
                }
                newOption.lmContextBuf = new int[newOption.lmContextBufLen];
                System.arraycopy(option.lmContextBuf, 0, newOption.lmContextBuf, 0,
                    newOption.lmContextBufLen - translation.english.indexedEnglish.length);
                System.arraycopy(translation.english.indexedEnglish, 0,
                    newOption.lmContextBuf,
                    newOption.lmContextBufLen - translation.english.indexedEnglish.length,
                    translation.english.indexedEnglish.length);
              }
              
              newOption.score = score;
              //beams[end].setPriority(newOption, score);
              beams[end].relaxPriority(newOption, score);
            }
          }
        }
      }
    }

    // System.out.println("Final score: " + scores[length - 1]);

    // Need to back trace.
    BeamSearchOption option = beams[length].getFirst();
    do {
      ret.add(0, option.phrasePair);
      option = option.prev;
    } while (option.prev != null);
    

    // double explicitScore = Decoder.StaticMethods.scoreHypothesis(ret, lm,
    // dm);
    // if (Math.abs(explicitScore - totalScore) > 1e-4) {
    // System.err.println("Warning: score calculated during decoding ("
    // + totalScore + ") does not match explicit scoring ("
    // + explicitScore + ")");
    // }
    
    //System.exit(0);
    
    return ret;
  }

  private double scoreLm(final int prevLmStateLength, final int[] lmStateBuf,
      final int totalTrgLength, final NgramLanguageModel lm) {
    double score = 0.0;

    if (prevLmStateLength < lmOrder - 1) {
      for (int i = 1; prevLmStateLength + i < lmOrder; ++i) {
        final double lmProb = lm.getNgramLogProbability(lmStateBuf, 0,
            prevLmStateLength + i);
        score += lmProb;
      }
    }

    for (int i = 0; i <= totalTrgLength - lmOrder; ++i) {
      final double lmProb = lm.getNgramLogProbability(lmStateBuf, i, i
          + lmOrder);
      score += lmProb;
    }
    return score;
  }

}
