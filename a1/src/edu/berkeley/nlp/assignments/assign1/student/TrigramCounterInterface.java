package edu.berkeley.nlp.assignments.assign1.student;

public interface TrigramCounterInterface {

   /**
    * Increase the count of the trigram.
    * 
    * @param w1w2
    *           The bigram index of the first two grams.
    * @param w3
    *           The unigram index of the third gram.
    */
   public abstract int increaseCount(int w1w2, int w3);

   public abstract void reportTopTrigram();
   
   public abstract int size();
   
   public abstract int get(int key1, int key2);
   
   public abstract int adjustOrPutValue(int key1, int key2, int adjustAmount);
   
   public abstract long sum();

}