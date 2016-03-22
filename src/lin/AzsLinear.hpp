/* * * * *
 *  AzsLinear.hpp 
 *  Copyright (C) 2013 Rie Johnson
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * * * * */

#ifndef _AZS_LINEAR_HPP_
#define _AZS_LINEAR_HPP_

#include "AzUtil.hpp"
#include "AzSmat.hpp"
#include "AzDmat.hpp"
#include "AzSvDataS.hpp"
#include "AzParam.hpp"
#include "AzHelp.hpp"

class AzsLinear {
protected:
  /*---  parameters  ---*/
  bool do_no_intercept, do_dense, do_sparse, do_regress;  
  AzBytArr s_train_x_fn, s_train_tar_fn, s_test_x_fn, s_test_tar_fn; 
  AzBytArr s_param; 
  
public://do_regress is set true, then regression is adopted. Otherwise, classification is adopted.   
  AzsLinear() : do_no_intercept(false), do_dense(false), do_sparse(false), do_regress(true) {}

  void train_test(const char *param); 
  void printHelp(const AzOut &out) const; 
  
protected:   
  void set_labels(const AzDvect *v_mc, AzIntArr &ia_lab) const; 
  bool is_dense_matrix(const AzSmat *m) const; 
  
  void resetParam(AzParam &azp); 
  void printParam(const AzOut &out) const; 
  void throw_if_empty(const char *kw, const AzBytArr &s) const {
    if (s.length() <= 0) throw new AzException(AzInputError, "AzsLinear::resetParam", kw, "must be specified."); 
  }  
}; 
#endif 
