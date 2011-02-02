package edu.berkeley.nlp.assignments.assign1.student;


/**
 * A very memory efficient trigram counter using a modified hash map
 * implementation. The hash map has only one long array (64bit each element).
 * Both the keys (trigram) and the values are packed into a single long
 * primitive to be stored.
 * 
 * The most frequent trigram in this corpus appears 468,261 times, costing 19
 * bits to store (the value).
 * 
 * To store the key (i.e. trigram), we need to store one unigram (495,172
 * unigrams or 19 bits) and a bigram index (8,374,230 bigrams or 23 bits).
 * 
 * In total, we need 19 + 19 + 23 = 61 bits to store!
 * 
 * @author rxin
 * 
 */
public class TrigramCounter extends CrazilyPackedHashMap implements
      TrigramCounterInterface {

   public TrigramCounter(int initialCapacity, float loadFactor) {
      super(initialCapacity, loadFactor, 0x3FFFFF, 23, 19);
   }

   /* (non-Javadoc)
    * @see edu.berkeley.nlp.assignments.assign1.student.TrigramCounterInterface#increaseCount(int, int)
    */
   @Override
   public int increaseCount(int w1w2, int w3) {
      return adjustOrPutValue(w1w2, w3, 1);
   }
   
   /* (non-Javadoc)
    * @see edu.berkeley.nlp.assignments.assign1.student.TrigramCounterInterface#reportTopTrigram()
    */
   @Override
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
