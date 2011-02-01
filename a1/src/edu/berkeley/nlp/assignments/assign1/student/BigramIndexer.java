package edu.berkeley.nlp.assignments.assign1.student;


public class BigramIndexer extends CrazilyPackedHashMap {

   public BigramIndexer(int initialCapacity, float loadFactor) {
      super(initialCapacity, loadFactor, 0x3FFFFFF, 19, 19);
   }

   /**
    * Adds a new word to the indexer.
    * 
    * @param w1
    *           First word in the bigram.
    * @param w2
    *           Second word in the bigram.
    *           
    * @return The index of bigram, starting at 0.
    */
   public int addAndGetIndex(int w1, int w2) {
      int index = get(w1, w2);
      if (index > 0) {
         return index;
      } else {
         adjustOrPutValue(w1, w2, size() + 1);
         return size();
      }
   }
}
