/* Copyright 2017 Andrey Goryachev

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

// Undef macros from tsl, so as to not polute the global namespace.
#ifdef HNSW_TSL_NO_RANGE_ERASE_WITH_CONST_ITERATOR
#undef HNSW_TSL_NO_RANGE_ERASE_WITH_CONST_ITERATOR
#endif

#ifdef hnsw_tsl_assert
#undef hnsw_tsl_assert
#endif
