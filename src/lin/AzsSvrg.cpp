/* * * * *
 *  AzsSvrg.cpp 
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
 
#include "AzsSvrg.hpp"
#include <string>
//#include <stdlib.h>
/*--------------------------------------------------------*/
void AzsSvrg::train_test_classif(const char *param, 
                         const AzDSmat *_m_trn_x, const AzIntArr *_ia_trn_lab, 
                         const AzDSmat *_m_tst_x, const AzIntArr *_ia_tst_lab, 
                         int _class_num)
{
  /*---  set data info into class variables so that everyone can see ... ---*/
  m_trn_x = _m_trn_x; ia_trn_lab = _ia_trn_lab; 
  m_tst_x = _m_tst_x; ia_tst_lab = _ia_tst_lab; 
  v_trn_y = v_tst_y = NULL; 
 
  class_num = _class_num; 
  if (class_num == 2) {
    AzTimeLog::print("Binary classification ... ", log_out); 
    class_num = 1; 
  }
  
  /*---  parse parameters  ---*/
  AzParam azp(param); 
  resetParam(azp); 
  printParam(log_out); 
  azp.check(log_out); 

  /*---  training and testing  ---*/  
  _train_test(); 
}

/*--------------------------------------------------------*/
void AzsSvrg::train_test_regress(const char *param, 
                         const AzDSmat *_m_trn_x, const AzDvect *_v_trn_y, 
                         const AzDSmat *_m_tst_x, const AzDvect *_v_tst_y)
{
  /*---  set data info into class variables so that everyone can see ... ---*/
  m_trn_x = _m_trn_x; v_trn_y = _v_trn_y; 
  m_tst_x = _m_tst_x; v_tst_y = _v_tst_y; 
  ia_trn_lab = ia_tst_lab = NULL; 
 
  class_num = 1; 
  AzTimeLog::print("Regression ... ", log_out); 
  AzTimeLog::print2Logfile("Regression ... \n","log.txt","a+");

  /*---  parse parameters  ---*/
  AzParam azp(param); 
  resetParam(azp); 
  printParam(log_out); 
  azp.check(log_out); 

  /*---  training and testing  ---*/
  _train_test(); 
}

/*--------------------------------------------------------*/
void AzsSvrg::_train_test()
{  
  if (rseed > 0) {
    srand(rseed); /* initialize the random seed */
  }

  /*---  initialization  ---*/
  int dim = m_trn_x->rowNum(); 
  reset_weights(dim); 

  /*---  iterate ... ---*/
  AzTimeLog::print("---  Training begins ... ", log_out); 
  AzsSvrgData_fast prev_fast; 
  AzsSvrgData_compact prev_compact; 
  int ite;
  string str1,str2,str3;
  for (ite = 0; ite < ite_num; ++ite) {
    char ite_count[4];
    sprintf(ite_count,"%d",ite);
    str1 = "\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>";
    str2 = "th iteration>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n";
    string str_temp_0 = str1+ite_count+str2;
    const char* log = str_temp_0.c_str();
    AzTimeLog::print2Logfile(log,"log.txt","a+");
    AzTimeLog::print2Logfile(log,"weights_log.txt","a+");

    if (do_show_timing) AzTimeLog::print("---  iteration#", ite+1, log_out); 
    if (doing_svrg(ite) && (ite-sgd_ite) % svrg_interval == 0) {
      if (do_show_timing) AzTimeLog::print("Computing gradient average ... ", log_out); 
      if (do_compact) get_avg_gradient_compact(&prev_compact); 
      else            get_avg_gradient_fast(&prev_fast); 
    }  

    if (do_show_timing) AzTimeLog::print("Updating weights ... ", log_out); 
    AzIntArr ia_dxs; 
    const int *dxs = gen_seq(dataSize(), ia_dxs); 
    /*******************************************************/
    //we add a speedup factor to the learning rate, and hope this can speedup the calculation
    double lam_speedup=0.0;
    lam_speedup=prev_fast.m_gavg.col(0).selfInnerProduct();
    lam=lam+lam_speedup;
    //time_t v_local_time;
    /********************************************************/
    int ix; 
    //for (ix = 0; ix < dataSize(); ++ix) {//original settings
    for(ix=0;ix<1000;ix++){// we observe that the loss will not decrease when the inner loop runs for 1000 rounds   
      int dx = dxs[ix];  /* data point index */
      AzDvect v_deriv(class_num); 
      get_deriv(dx, &v_deriv); /* compute the derivatives */
      if (doing_svrg(ite)) {
        if (do_compact) updateDelta_svrg_compact(dx, &v_deriv, prev_compact); 
        else            updateDelta_svrg_fast(dx, &v_deriv, prev_fast); 
      }
      else {
        updateDelta_sgd(dx, &v_deriv);       
      } 
      /*log each update and the value of loss function*/
      double loss_1;/*this is the value of loss function which does not include the regularization*/
      double loss = get_loss_regress(m_trn_x, v_trn_y, &loss_1);
      char itnum[32],loss_str[32];
      sprintf(itnum,"%d",ix);
      sprintf(loss_str,"%f",loss);
      str1 = "\n#############   Inner loop:";
      str2 = "   the value of loss function is:";
      str3 = "\n";
      string str_temp = str1+itnum+str2+loss_str+str3;
      const char* log2 = str_temp.c_str();
      AzTimeLog::print2Logfile(log2,"log.txt","a+");
      str1 = "\n>Inner loop:";
      str2 = ":\n";
      string str0 = str1 +itnum+str2;
      const char* log_weights = str0.c_str();
      AzTimeLog::print2Logfile(log_weights,"weights_log.txt","a+");
      const int dim = m_trn_x->rowNum();
      double* weights = m_w.get(0);
      AzTimeLog::printDouble2Logfile(weights, dim,"weights_log.txt","a+");
      flushDelta(); 
    }
    //show_perf(ite); 
  }

  if (do_show_timing) AzTimeLog::print("--- End of training ... ", log_out); 
  
  /*---  write predictions to a file if requested  ---*/
  //if (s_pred_fn.length() > 0) {
  //  AzTimeLog::print("Writing predictions to ", s_pred_fn.c_str(), log_out); 
  //  write_pred(m_tst_x, s_pred_fn.c_str()); 
  //}
}

/*--------------------------------------------------------*/
void AzsSvrg::reset_weights(int dim) {
  AzsLmod::reset_weights(dim, class_num); 
  m_w_delta.reform(dim, class_num); 
  m_w_delta_prev.reform(dim, class_num); 
}

/*--------------------------------------------------------*/
void AzsSvrg::updateDelta_sgd(int dx, 
                              const AzDvect *v_deriv)
{
  int cx;
  for (cx = 0; cx < class_num; ++cx) {
    /*---  current gradient  ---*/
    m_trn_x->add_to(m_w_delta.col_u(cx), dx, v_deriv->get(cx)); /* w_delta[,col] += trn_x[,dx]*deriv[cx] */ 
  }
}

/*------------------------------------------------------------------------*/
/* fast version (default): to compute the gradient with previous weights, */
/* do "gradient <- x times derivative" instead of recomputing it.         */
/*------------------------------------------------------------------------*/
void AzsSvrg::updateDelta_svrg_fast(int dx, 
                                    const AzDvect *v_deriv, 
                                    const AzsSvrgData_fast &prev)
{
  int cx;
  for (cx = 0; cx < class_num; ++cx) {
    /*---  add current gradient - previous gradient  ---*/
    double coeff = v_deriv->get(cx) - prev.m_deriv.get(cx, dx); /* coeff = deriv[cx] - prev.deriv[cx,dx] */  
    m_trn_x->add_to(m_w_delta.col_u(cx), dx, coeff);  /* w_delta[,cx] += trn_x[,dx]*coeff */ 
  }
  
  /*---  add average gradient  ---*/
  m_w_delta.add(&prev.m_gavg); /* w_delta += gavg */
}

/*------------------------------------------------------------------------------*/ 
/* fast version (default): keep the derivatives (scalars) with previous weights */
/* so that gradients with previous weights do not have to be re-computed.       */
/* This is faster at the expense of using more memory.                          */
/*------------------------------------------------------------------------------*/ 
void AzsSvrg::get_avg_gradient_fast(AzsSvrgData_fast *sd) /* output */
const
{
  int data_size = m_trn_x->colNum(); 
  sd->m_deriv.reform(class_num, data_size); /* derivatives */
  sd->m_gavg.reform(m_trn_x->rowNum(), class_num); /* gradient average */
  int dx;
  for (dx = 0; dx < data_size; ++dx) {
    get_deriv(dx, sd->m_deriv.col_u(dx)); 
    int cx; 
    for (cx = 0; cx < class_num; ++cx) {
      double my_deriv = sd->m_deriv.get(cx, dx); 
      /*---  add the gradient  ---*/   
      m_trn_x->add_to(sd->m_gavg.col_u(cx), dx, my_deriv);  /* gavg[,cs] += trn_x[,dx]*my_deriv */          
    }
  }
  sd->m_gavg.divide(data_size);  /* take the average */
}

/*----------------------------------------------------------------*/
/* compact version: recompute the gradient with previous weights  */
/* at every iteration.                                            */
/*----------------------------------------------------------------*/
void AzsSvrg::updateDelta_svrg_compact(int dx, 
                                       const AzDvect *v_deriv, 
                                       const AzsSvrgData_compact &prev)
{
  /*---  compute derivatives with the previous weights  ---*/
  AzDvect v_prev_deriv(class_num); 
  get_deriv(&prev.lmod, dx, &v_prev_deriv);   
  int cx;
  for (cx = 0; cx < class_num; ++cx) {
    /*---  add current gradient - previous gradient  ---*/
    double coeff = v_deriv->get(cx) - v_prev_deriv.get(cx); /* coeff = deriv[cx] - prev_deriv[cx] */
    m_trn_x->add_to(m_w_delta.col_u(cx), dx, coeff);  /* w_delta[,cx] += x*coeff */   
  }
  /*---  add average gradient  ---*/
  m_w_delta.add(&prev.m_gavg); /* w_delta += gavg */
}

/*-----------------------------------------------------------------------*/ 
/* compact version: don't keep the derivatives with previous weights.    */
/* This is slower but uses less memory.                                  */
/*-----------------------------------------------------------------------*/ 
void AzsSvrg::get_avg_gradient_compact(AzsSvrgData_compact *sd) /* output */
const
{
  int data_size = m_trn_x->colNum(); 
  sd->m_gavg.reform(m_trn_x->rowNum(), class_num); /* gradient average */
  AzDvect v_deriv(class_num); 
  int dx;
  for (dx = 0; dx < data_size; ++dx) {
    get_deriv(dx, &v_deriv); 
    int cx; 
    for (cx = 0; cx < class_num; ++cx) {
      double my_deriv = v_deriv.get(cx); 
      /*---  add the gradient  ---*/
#if 0       
      sd->m_gavg.col_u(cx)->add(m_trn_x->col(dx), my_deriv); /* gavg[,cs] += trn_x[,dx]*my_deriv */
#else 
      m_trn_x->add_to(sd->m_gavg.col_u(cx), dx, my_deriv); /* gavg[,cs] += trn_x[,dx]*my_deriv */
#endif 
    }
  }
  sd->m_gavg.divide(data_size);  /* take the average */
  sd->lmod.reset(&m_w, ws); /* save the weights */
}

/*------------------------------------------------------------*/  
/*------------------------------------------------------------*/  
void AzsSvrg::flushDelta() 
{
  if (momentum > 0) { /* use momentum */
    check_ws("flushDelta with momentum"); 
    m_w_delta.multiply(-eta); 
    m_w_delta.add(&m_w, -eta*lam); 
    m_w_delta.add(&m_w_delta_prev, momentum); 
    m_w.add(&m_w_delta); 
    m_w_delta_prev.set(&m_w_delta); 
  }
  else { /* don't use momentum */
    ws *= (1 - lam*eta);  
    m_w.add(&m_w_delta, -eta/ws);   
  }
  m_w_delta.zeroOut(); 
  
  if (ws < 1e-10) {
    flush_ws(); 
  }
}         

/*------------------------------------------------------------*/  
double AzsSvrg::regloss(int col) const {
  if (col < 0) return 0.5*lam*m_w.squareSum()*ws*ws; 
  else         return 0.5*lam*m_w.col(col)->squareSum()*ws*ws; 
}

/*------------------------------------------------------------*/ 
void AzsSvrg::show_perf(int ite) const
{
  int myite = ite+1 - sgd_ite; 
  if (ite+1 == ite_num || /* end of training */
      myite >= 0 && test_interval > 0 && myite%test_interval == 0) {
    if (do_show_timing) AzTimeLog::print("Testing ... ", log_out); 
    if (doing_classif()) show_perf_classif(ite);   
    else                 show_perf_regress(ite); 
  }
} 

/*------------------------------------------------------------*/ 
void AzsSvrg::show_perf_classif(int ite) const
{
  AzBytArr s("ite,");s.cn(ite+1);
  if (do_show_loss) {
    double loss_woreg, test_loss; 
    double tr_loss = get_loss_classif(m_trn_x, ia_trn_lab, &loss_woreg); 
    get_loss_classif(m_tst_x, ia_tst_lab, &test_loss); 
    s.c(",training-loss,");s.cn(tr_loss,15);
    /* s.c(",");s.cn(loss_woreg,15); */
    s.c(",test-loss,");s.cn(test_loss,15); 
  }
  double acc = test_classif(m_tst_x, ia_tst_lab); 
  s.c(",errrate,");s.cn(1-acc);   
  AzPrint::writeln(log_out, s); 
}    

/*------------------------------------------------------------*/ 
void AzsSvrg::show_perf_regress(int ite) const
{
  AzBytArr s("ite,");s.cn(ite+1);
  double loss_woreg, test_loss; 
  if (do_show_loss) {
    double tr_loss = get_loss_regress(m_trn_x, v_trn_y, &loss_woreg); 
    s.c(",training-loss,");s.cn(tr_loss,15);
    /* s.c(",");s.cn(loss_woreg,15); */
  }
  get_loss_regress(m_tst_x, v_tst_y, &test_loss); 
  s.c(",test-loss,");s.cn(test_loss,15); 
  AzPrint::writeln(log_out, s); 
} 

/*--------------------------------------------------------*/
void AzsSvrg::get_deriv(const AzsLmod *mod, 
                        int dx,  /* input: data index */
                        AzDvect *v_deriv) /* output: derivatives */
const                              
{
  if (doing_classif()) {
    int cls = ia_trn_lab->get(dx); 
    mod->get_deriv_classif(loss_type, m_trn_x, dx, cls, v_deriv); 
  }
  else {
    double y = v_trn_y->get(dx); 
    mod->get_deriv_regress(loss_type, m_trn_x, dx, y, v_deriv); 
  }
}
			
/*--------------------------------------------------------*/
double AzsSvrg::get_loss_classif(const AzDSmat *m_x, const AzIntArr *ia_lab, double *out_loss1) const 
{
  double loss = getLossAvg_classif(loss_type, m_x, ia_lab); 
  if (out_loss1 != NULL) *out_loss1 = loss; 
  loss += regloss(); 
  return loss; 
}  

/*--------------------------------------------------------*/
double AzsSvrg::get_loss_regress(const AzDSmat *m_x, const AzDvect *v_y, double *out_loss1) const 
{
  double loss = getLossAvg_regress(loss_type, m_x, v_y); 
  if (out_loss1 != NULL) *out_loss1 = loss; 
  loss += regloss(); 
  return loss; 
}

/*--------------------------------------------------------*/
#define kw_loss "loss="
#define kw_svrg_interval "svrg_interval="
#define kw_sgd_ite "sgd_iterations="
#define kw_test_interval "test_interval="

#define kw_ite_num     "num_iterations="
#define kw_momentum    "momentum="
#define kw_rseed       "random_seed="
#define kw_eta         "learning_rate="
#define kw_lam      "lam="  
#define kw_do_compact "Compact"
#define kw_do_show_loss "ShowLoss"
#define kw_do_show_timing "ShowTiming"
#define kw_pred_fn "prediction_fn="
#define kw_with_replacement "WithReplacement"
/*--------------------------------------------------------*/
void AzsSvrg::resetParam(AzParam &azp)
{
  AzBytArr s_loss; 
  azp.vStr(kw_loss, &s_loss); 
  if (s_loss.length() > 0) loss_type = lossType(s_loss.c_str()); 
  azp.vInt(kw_svrg_interval, &svrg_interval); 
  azp.vInt(kw_sgd_ite, &sgd_ite); 
  azp.vInt(kw_test_interval, &test_interval); 

  azp.vInt(kw_ite_num, &ite_num); 
  azp.vInt(kw_rseed, &rseed); 
  azp.vFloat(kw_eta, &eta);
  azp.vFloat(kw_momentum, &momentum); 
  azp.vFloat(kw_lam, &lam); 
  azp.swOn(&do_compact, kw_do_compact); 
  azp.swOn(&do_show_loss, kw_do_show_loss); 
  azp.swOn(&do_show_timing, kw_do_show_timing); 
  azp.swOn(&with_replacement, kw_with_replacement); 
  azp.vStr(kw_pred_fn, &s_pred_fn); 
}

/*--------------------------------------------------------*/
void AzsSvrg::printParam(const AzOut &out) const 
{
  if (out.isNull()) return; 

  AzPrint o(out); 
  o.reset_options(); 
  o.set_precision(5); 
  o.ppBegin("", ""); 
  o.printV(kw_loss, lossName(loss_type)); 
  o.printV(kw_svrg_interval, svrg_interval); 
  o.printV(kw_sgd_ite, sgd_ite);
  o.printV(kw_test_interval, test_interval);  

  o.printV(kw_ite_num, ite_num); 
  o.printV(kw_rseed, rseed); 
  o.printV(kw_eta, eta); 
  o.printV(kw_momentum, momentum); 
  o.printV(kw_lam, lam);     
  o.printSw(kw_do_compact, do_compact); 
  o.printSw(kw_do_show_loss, do_show_loss); 
  o.printSw(kw_do_show_timing, do_show_timing); 
  o.printSw(kw_with_replacement, with_replacement); 
  o.printV_if_not_empty(kw_pred_fn, s_pred_fn); 
  
  o.ppEnd(); 

  /*---  check parameters  ---*/  
  throw_if_nonpositive(kw_ite_num, (double)ite_num); 
  throw_if_nonpositive(kw_eta, eta); 
  throw_if_negative(kw_lam, lam); 
  throw_if_nonpositive(kw_svrg_interval, svrg_interval); 
  throw_if_negative(kw_sgd_ite, sgd_ite); 
  if (test_interval > 0) {
    if (test_interval % svrg_interval != 0) {
      AzBytArr s(kw_test_interval); s.c(" must be a multiple of "); s.c(kw_svrg_interval); 
      throw new AzException(AzInputError, "AzsSvrg::resetParam", s.c_str()); 
    }
  }
  if (!doing_classif() && loss_type != AzsLoss_Square && loss_type != AzsLoss_Logistic) {//regression task!NOTICE:we have change it
    throw new AzException(AzInputError, "AzsSvrg::resetParam", "Only square or Logistic loss is supported for regression tasks.");
    //throw new AzException(AzInputError, "AzsSvrg::resetParam", "Only square loss is supported for regression tasks."); 
  }
  if (loss_type == AzsLoss_None) {
    AzBytArr s(kw_loss); s.c(" must be specified.  "); allLoss_str(s); 
    throw new AzException(AzInputError, "AzsSvrg::resetParam", s.c_str()); 
  }
}

/*------------------------------------------------------------------*/
/* static */
void AzsSvrg::printHelp(AzHelp &h)
{ 
  h.item_required(kw_ite_num, "Number of iterations (i.e., how many times to go through the training data).", " 30"); 
  h.item_required(kw_svrg_interval, "SVRG interval.  E.g., if this value is 2, average gradient is computed after 2 iterations, 4 iterations, and so on.  Note: one iteration goes through the entire training data once.");   
  h.item_required(kw_sgd_ite, "number of initial SGD iterations before starting SVRG."); 
  h.item_required(kw_eta, "Learning rate."); 
  h.item_required(kw_lam, "L2 regularization parameter."); 
  h.item_required(kw_loss, "Loss function.  Logistic | Square", " Logistic"); 
  h.item_noquotes("", "\"Logistic\" with >2 classes: multi-class logistic; one vs. all otherwise.  Use \"Square\" if the task is regression.");   
  h.nl(); 
  h.item_experimental(kw_momentum, "Momentum"); 
  h.item(kw_pred_fn, "File to write predictions at the end of training.  Optional");          
  h.item(kw_rseed, "Seed for randomizing the order of training data points.", " 1"); 
  h.item_experimental(kw_with_replacement, "Randomize the order of training data points with replacement."); 
  h.item(kw_test_interval, "How often to test.  E.g., if this value is 2, test is done after 2 iterations, 4 iterations, and so on.  It must be a multiple of svrg_interval.", 
         " once at the end of training"); 
  h.item(kw_do_compact, "When specified, derivatives with previous weights are not saved and recomputed, which consumes a little less memory and slows down the training a little."); 
  h.item(kw_do_show_loss, "Show training loss (training objective including the regularization term) and test loss when test is done.  If \"Regression\" is on, test loss is always shown irrespective of this switch."); 
  h.item(kw_do_show_timing, "Display time stamps to show progress."); 
}

/*------------------------------------------------------------------*/
const int *AzsSvrg::gen_seq(int data_size, AzIntArr &ia) const
{
  ia.range(0, data_size); 
  AzTools::shuffle(-1, &ia, with_replacement); 
  return ia.point(); 
}
