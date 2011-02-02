package edu.berkeley.nlp.assignments.assign1.student;


public class TrigramCounterApproximate extends CrazilyPackedHashMapApproximate
      implements TrigramCounterInterface {

   public TrigramCounterApproximate(int initialCapacity, float loadFactor) {
      super(initialCapacity, loadFactor, 0x3FFFFF, 23);
   }

   /**
    * Increase the count of the trigram.
    * 
    * @param w1w2
    *           The bigram index of the first two grams.
    * @param w3
    *           The unigram index of the third gram.
    */
   public int increaseCount(int w1w2, int w3) {
      return adjustOrPutValue(w1w2, w3, 1);
   }
   
   public void reportTopTrigram() {
      long maxCount = 0;
      for (int i = 0; i < data.length; i++) {
         long value = data[i] & valueMask;
         //int key = data[i] & keyMask;
         if (value > maxCount) maxCount = value;
      }
      System.out.println("Top trigram count: " + maxCount);
   }
   
}
