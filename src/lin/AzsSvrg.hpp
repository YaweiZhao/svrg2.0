/* * * * *
 *  AzsSvrg.hpp 
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

#ifndef _AZS_SVRG_HPP_
#define _AZS_SVRG_HPP_

#include "AzUtil.hpp"
#include "AzSmat.hpp"
#include "AzDmat.hpp"
#include "AzParam.hpp"
#include "AzHelp.hpp"
#include "AzsLmod.hpp"

class AzsSvrgData_fast {
public:
  AzDmat m_gavg, m_deriv; 
}; 

class AzsSvrgData_compact {
public:
  AzDmat m_gavg; 
  AzsLmod lmod; 
};

class AzsSvrg : public virtual /* extends */ AzsLmod {
protected:
  const AzDSmat *m_trn_x, *m_tst_x; /* pointers to the data point matrices */
  const AzIntArr *ia_trn_lab, *ia_tst_lab;  /* pointers to the labels */
  const AzDvect *v_trn_y, *v_tst_y; 
  
  int class_num;      /* # of classes */
  AzBytArr s_pred_fn; /* filename to save the predictions at the end */
  
  AzDmat m_w_delta;
  AzDmat m_w_delta_prev; 

  /*---  Parameters  ---*/
  /*---  Note: for all numerical values, -1 means "no value" */
  double lam;      /* L2 regularization parameter */
  double eta;      /* learning rate */
  double momentum; /* momentum */
  AzsLossType loss_type; /* loss function */
  int svrg_interval; /* interval of computing average gradient */
  int sgd_ite;       /* # of sgd iterations to initialize weights */
  int test_interval; /* how frequently test should be done */
  int ite_num;       /* # of iterations */
  int rseed;         /* random number generator seed */
  bool with_replacement; 
  bool do_compact; /* true: derivatives with previous weights are not saved */
  bool do_show_loss, do_show_timing; 
  
public:   
  AzsSvrg() : m_trn_x(NULL), m_tst_x(NULL), ia_trn_lab(NULL), ia_tst_lab(NULL), 
              v_trn_y(NULL), v_tst_y(NULL), class_num(-1), 
              lam(-1), eta(-1), loss_type(AzsLoss_None), 
              svrg_interval(-1), sgd_ite(-1), test_interval(-1), 
              ite_num(-1), momentum(-1), rseed(1), 
              with_replacement(false), do_compact(false), do_show_loss(false), do_show_timing(false) {}

  void train_test_classif(const char *param, 
                  const AzDSmat *_m_trn_x, const AzIntArr *_ia_trn_lab, /* training data */
                  const AzDSmat *_m_tst_x, const AzIntArr *_ia_tst_lab, /* test data */
                  int _class_num); 

  void train_test_regress(const char *param, 
                  const AzDSmat *_m_trn_x, const AzDvect *v_trn_y,  /* training data */
                  const AzDSmat *_m_tst_x, const AzDvect *v_tst_y); /* test data */
  static void printHelp(AzHelp &h); 
                  
protected:   
  void _train_test(); 
  int dataSize() const { return (m_trn_x == NULL) ? 0 : m_trn_x->colNum(); }
  bool doing_svrg(int ite) const { return (svrg_interval > 0 && ite >= sgd_ite) ? true : false; }
  bool doing_classif() const { return (ia_trn_lab != NULL); }
  
  void reset_weights(int dim); 

  void updateDelta_sgd(int dx, const AzDvect *v_deriv); 
  void flushDelta(); 
  double regloss(int col=-1) const; 
  
  void show_perf(int ite) const; 
  void show_perf_classif(int ite) const; 
  void show_perf_regress(int ite) const; 
  
  /*---  fast svrg (save derivatives)  ---*/  
  void updateDelta_svrg_fast(int dx, const AzDvect *v_deriv, const AzsSvrgData_fast &prev); 
  void get_avg_gradient_fast(AzsSvrgData_fast *sd) const; /* output */

  /*---  compact svrg (don't save derivatives)  ---*/  
  void updateDelta_svrg_compact(int dx, const AzDvect *v_deriv, const AzsSvrgData_compact &prev); 
  void get_avg_gradient_compact(AzsSvrgData_compact *sd) const; /* output */
  
  /*---  ---*/
  inline void get_deriv(int dx, AzDvect *v_deriv) const {
    get_deriv(this, dx, v_deriv); 
  }
  void get_deriv(const AzsLmod *mod, /* model */
                 int dx, 
                 /*---  output  ---*/
                 AzDvect *v_deriv) const; /* must be formatted by caller */

  /*--------------------------------------------------------*/
  double get_loss_classif(const AzDSmat *m_x, const AzIntArr *ia_lab, double *out_loss1) const; 
  double get_loss_regress(const AzDSmat *m_x, const AzDvect *v_y, double *out_loss1) const; 
  /*--------------------------------------------------------*/
  
  void resetParam(AzParam &azp); 
  void printParam(const AzOut &out) const; 
  template <class T>
  void throw_if_nonpositive(const char *kw, T val) const {
    if (val <= 0) throw new AzException(AzInputError, "AzsSvrg::resetParam", kw, "must be positive."); 
  }
  template <class T>
  void throw_if_negative(const char *kw, T val) const {
    if (val < 0) throw new AzException(AzInputError, "AzsSvrg::resetParam", kw, "must be non-negative."); 
  }
  const int *gen_seq(int data_size, AzIntArr &ia) const; 
}; 
#endif 
