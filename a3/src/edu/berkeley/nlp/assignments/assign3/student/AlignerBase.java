package edu.berkeley.nlp.assignments.assign3.student;

import edu.berkeley.nlp.langmodel.EnglishWordIndexer;
import edu.berkeley.nlp.mt.Alignment;
import edu.berkeley.nlp.mt.SentencePair;
import edu.berkeley.nlp.mt.WordAligner;
import edu.berkeley.nlp.util.CounterMap;
import edu.berkeley.nlp.util.StringIndexer;

/**
 * Base class for aligners.
 * 
 * @author rxin
 */
public abstract class AlignerBase implements WordAligner {

   protected static final int MAX_SENTENCE_LEN = 1024;
   
   /**
    * French word indexer that maps a French word (java string) to an integer.
    */
   protected StringIndexer frenchWordIndexer = new StringIndexer();
   
   /**
    * English word indexer that maps an English word (java string) to an
    * integer.
    */
   protected StringIndexer englishWordIndexer = EnglishWordIndexer.getIndexer();
   
   protected DynamicIntArray englishWordCounter = new DynamicIntArray(1000);
   
   protected DynamicIntArray foreignWordCounter = new DynamicIntArray(1000);
   
   /**
    * Translation probability for <English word, French word> pair occurs.
    */
   protected CounterMap<Integer, Integer> pairCounters =
      new CounterMap<Integer, Integer>();
   
   /**
    * Train the aligner. This must be called before alignSentencePair().
    * @param trainingData
    */
   public abstract void train(Iterable<SentencePair> trainingData);
   
   /* (non-Javadoc)
    * @see edu.berkeley.nlp.mt.WordAligner#alignSentencePair(edu.berkeley.nlp.mt.SentencePair)
    */
   public abstract Alignment alignSentencePair(SentencePair sentencePair);
   
}