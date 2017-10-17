// Undef macros from tsl, so as to not polute the global namespace.
#ifdef HNSW_TSL_NO_RANGE_ERASE_WITH_CONST_ITERATOR
#undef HNSW_TSL_NO_RANGE_ERASE_WITH_CONST_ITERATOR
#endif

#ifdef hnsw_tsl_assert
#undef hnsw_tsl_assert
#endif
