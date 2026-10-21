"""Original S1/M0_FORMAL input preparation, never a new feature producer.

Pure frame APIs support synthetic checks. The activated entry requires an
external registered executor and identity-bound original-input loader; this
module has no raw-data CLI, network, registry, model fit or inference path.
"""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import inspect
from pathlib import Path
import textwrap

import environment_tilt_m0_oof as support

FLAGS = support.FLAGS
BATCH = support.BATCH
FEATURES = support.FEATURES
POLICY = dict(output_start='2020-01-01',output_end='2023-12-31',
    producer='ORIGINAL_S1_MATERIALIZE_M0_FORMAL_FIVE_ARM_COMMON_FINITE',
    warmup='ALL_CERTIFIED_AVAILABLE_PER_STOCK_HISTORY_BEFORE_OUTPUT_PROJECTION',
    calendar='EXTENDED_Q_DATES_UNION_ORIGINAL_BOUND_CALENDAR',
    index='UNCHANGED_OBSERVED_VALUES_REQUIRED_OUTPUT_AND_SOURCE_DERIVED_LOOKBACK',
    old_overlap='EXACT_KEYS_ORDER_AND_36_FLOAT64_BYTES',
    model_protection='ALL_CONFIGURED_ACCEPTED_TRAIN_TEST_DATA_IDENTITIES',
    on_parity_failure='INPUT_VERSION_PARITY_FAILED_NO_MIX_OR_REFIT',**FLAGS)
POLICY_SHA256 = support.digest(POLICY)


class Blocked(ValueError):pass


class NativeBudget:
    """Reuse the accepted native Mac peak-RSS/memory/disk probe, no psutil."""
    def __init__(self,config):self.config=config
    def check(self,stage):
        return support.resource_snapshot(dict(output_root=self.config['output_root'],
            resource_probe_source=self.config['sources']['resource_probe'],resource_caps=self.config['caps']))


def _dates(values):
    import pandas as pd
    d=pd.to_datetime(values)
    if d.isna().any() or getattr(d.dt,'tz',None) is not None or not d.equals(d.dt.normalize()):
        raise Blocked('EXACT_NORMALIZED_NAIVE_DATES_REQUIRED')
    return d


def _date_strings(values):return [v.date().isoformat() for v in values]


def compose_history(recipe,prefix_raw,old_raw,*,prefix_map,old_map,history_proof,max_rows):
    """Separate units, keep every old row, require certified full pre-old history."""
    import pandas as pd
    if len(prefix_raw)+len(old_raw)>max_rows:raise Blocked('INPUT_ROW_RESOURCE_BOUND')
    old=recipe.normalize_price_panel(old_raw,old_map,amount_multiplier=10000.)
    prefix=recipe.normalize_price_panel(prefix_raw,prefix_map,amount_multiplier=1.)
    if tuple(old.columns)!=tuple(prefix.columns) or old.empty:raise Blocked('NORMALIZED_PRICE_SCHEMA_REQUIRED')
    if history_proof.get('status')!='CERTIFIED_FULL_AVAILABLE_PRE_OLD_HISTORY':
        raise Blocked('FULL_AVAILABLE_WARMUP_HISTORY_NOT_PROVEN')
    if history_proof.get('prefix_amount_unit')!='CNY' or history_proof.get('old_amount_unit')!='10000_CNY' or history_proof.get('volume_unit')!='SHARES':
        raise Blocked('EXPLICIT_ORIGINAL_UNIT_BRIDGE_REQUIRED')
    first=old.groupby('code').date.min()
    if set(prefix.code)-set(first.index):raise Blocked('PREFIX_ADDS_UNBOUND_OLD_UNIVERSE_CODE')
    if any(t>=first[c] for t,c in prefix[['date','code']].itertuples(index=False,name=None)):
        raise Blocked('PREFIX_MUST_END_BEFORE_EACH_OLD_FIRST_ROW_NO_OVERWRITE')
    records=history_proof.get('by_code',{})
    if set(records)!=set(first.index):raise Blocked('ALL_OLD_CODES_REQUIRE_HISTORY_OR_NO_EARLIER_HISTORY_PROOF')
    for code,start in first.items():
        p=prefix.loc[prefix.code.eq(code)]
        r=records[code]
        if (r.get('old_start')!=start.date().isoformat()
                or r.get('prefix_rows')!=len(p)
                or r.get('prefix_dates_sha256')!=support.digest(_date_strings(p.date))
                or r.get('available_start')!=(p.date.min().date().isoformat() if len(p) else start.date().isoformat())
                or r.get('earlier_source_history_omitted') is not False
                or not r.get('source_evidence_sha256')):
            raise Blocked('PER_CODE_FULL_HISTORY_PROOF_CONFLICT')
    combined=pd.concat([prefix,old],ignore_index=True).sort_values(['date','code'],kind='mergesort').reset_index(drop=True)
    if combined.duplicated(['date','code']).any():raise Blocked('COMPOSED_HISTORY_DUPLICATE_KEYS')
    retained=combined.merge(old[['date','code']],on=['date','code'],how='inner',validate='one_to_one')
    pd.testing.assert_frame_equal(retained.reset_index(drop=True),old,check_exact=True)
    return combined,old,dict(prefix_rows=len(prefix),old_rows=len(old),composed_rows=len(combined),
        old_rows_retained_exactly=True,old_raw_mutated=False,
        currency_bridge='ORIGINAL_NORMALIZE_PREFIX_X1_OLD_X10000_TO_CNY',
        volume='SHARES_UNCHANGED',history_proof_sha256=support.digest(history_proof))


def extend_calendar(recipe,panel,old_panel,old_calendar):
    import numpy as np
    import pandas as pd
    old_cal=old_calendar.copy();old_cal['date']=_dates(old_cal.date)
    # Reuse original calendar validator, no strategy/event/label file read.
    old_cal=recipe.normalize_calendar(old_cal.rename(columns={'date':'trade_date'}),old_panel)
    old_q=set(old_panel.date);new_q=set(panel.date)
    if not old_q.issubset(new_q):raise Blocked('OLD_Q_DATES_NOT_RETAINED')
    values=sorted(new_q|set(old_cal.date))
    raw=pd.DataFrame(dict(trade_date=values,calendar_index=np.arange(len(values))))
    cal=recipe.normalize_calendar(raw,panel)
    overlap=cal.loc[cal.date.between(old_cal.date.min(),old_cal.date.max())].reset_index(drop=True)
    if tuple(overlap.date)!=tuple(old_cal.date):raise Blocked('OLD_OVERLAP_CALENDAR_SEQUENCE_DRIFT')
    shift=overlap.calendar_index.to_numpy()-old_cal.calendar_index.to_numpy()
    if len(set(shift))!=1:raise Blocked('OLD_CALENDAR_INDEX_SHIFT_NOT_CONSTANT')
    # Original lag-date mapping over old-defined predecessor sessions is exact.
    for lag in (1,5,20):
        if tuple(overlap.date.iloc[:-lag])!=tuple(old_cal.date.iloc[:-lag]):
            raise Blocked('OLD_SESSION_LAG_DATE_MAPPING_DRIFT')
    return cal,dict(rule=POLICY['calendar'],old_calendar_dates=len(old_cal),
        extended_calendar_dates=len(cal),old_index_constant_shift=int(shift[0]),
        old_overlap_sequence_exact=True,old_defined_lag_date_mapping_exact=True,
        earlier_predecessors_previously_unavailable='KEPT_AS_NEW_WARMUP_NOT_CLAIMED_OLD_VALUE_PARITY',
        official_calendar='CANNOT_CONFIRM')


def derive_index_lookback(recipe,formula_adapter,engine,v192_text,v193_text):
    """Read dependency ASTs, never evaluate or replace original expressions.

    None means independent of INDEXC. Recursive/dynamic index dependencies are
    rejected rather than assigned an invented finite warmup.
    """
    def formula_lags(text,targets):
        defs={name:expr for name,_,expr in formula_adapter.parse_definitions(text)}
        cache={};visiting=set()
        def name_lag(name):
            if name=='INDEXC':return 0
            if name not in defs:return None
            if name in visiting:raise Blocked('CYCLIC_FORMULA_DEPENDENCY')
            if name not in cache:
                visiting.add(name);cache[name]=node_lag(ast.parse(engine.tdx_expr_to_py(defs[name]),mode='eval').body);visiting.remove(name)
            return cache[name]
        def maximum(values):
            values=[v for v in values if v is not None]
            return max(values) if values else None
        def node_lag(node):
            if isinstance(node,ast.Name):return name_lag(node.id.upper())
            if isinstance(node,ast.Constant):return None
            if isinstance(node,ast.Call):
                if not isinstance(node.func,ast.Name):raise Blocked('UNKNOWN_FORMULA_CALL')
                name=node.func.id.upper();args=[node_lag(v) for v in node.args];base=maximum(args)
                if base is None:return None
                if name in ('EMA','SMA','BARSLAST'):raise Blocked('UNBOUNDED_INDEX_DEPENDENCY_REQUIRES_RULING')
                if name in ('MA','HHV','LLV','SUM','COUNT','EXIST','EVERY','HHVBARS','REF'):
                    if len(node.args)!=2 or not isinstance(node.args[1],ast.Constant) or type(node.args[1].value) is not int:
                        raise Blocked('DYNAMIC_INDEX_LOOKBACK_UNSUPPORTED')
                    window=node.args[1].value
                    if window<0 or (name!='REF' and window==0):raise Blocked('UNBOUNDED_OR_FUTURE_INDEX_LOOKBACK')
                    return base+window-(0 if name=='REF' else 1)
                if name=='CROSS':return base+1
                if name in ('IF','MAX','MIN','ABS','_NOT_'):return base
                raise Blocked('UNRECOGNIZED_INDEX_DEPENDENT_FUNCTION')
            return maximum(node_lag(v) for v in ast.iter_child_nodes(node))
        result={name:name_lag(name) for name in targets}
        return {k:v for k,v in result.items() if v is not None}
    v193=formula_lags(v193_text,(*formula_adapter.CORE_OUTPUTS,'XG','AUDITM'))
    v192=formula_lags(v192_text,('XG',))
    tree=ast.parse(textwrap.dedent(inspect.getsource(recipe.build_m1_features)))
    uses=[]
    for node in ast.walk(tree):
        if (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                and isinstance(node.func.value,ast.Name) and node.func.value.id=='index'
                and node.func.attr in ('rolling','pct_change')):
            if not node.args or not isinstance(node.args[0],ast.Constant) or type(node.args[0].value) is not int:
                raise Blocked('DYNAMIC_ORIGINAL_INDEX_ARM_LOOKBACK')
            n=node.args[0].value
            if n<=0:raise Blocked('INVALID_ORIGINAL_INDEX_ARM_LOOKBACK')
            uses.append(dict(operation=node.func.attr,window=n,previous_sessions=n-(node.func.attr=='rolling')))
    if not uses:raise Blocked('ORIGINAL_INDEX_ARM_LOOKBACK_NOT_LOCATED')
    return dict(calendar_previous_sessions=max(r['previous_sessions'] for r in uses),
        per_stock_previous_rows=max([*v193.values(),*v192.values(),0]),
        source_m1_uses=uses,v193_index_dependent_outputs=v193,v192_index_dependent_outputs=v192,
        source_derived=True,index_filled=False,
        original_formula_internal_fallback='UNCHANGED_ORIGINAL_IF_INDEXC_GT0_INDEXC_ELSE_C_NOT_AN_ADAPTER_FILL')


def required_index_dates(panel,calendar,lookback,output_start,*,through=None):
    import pandas as pd
    start=pd.Timestamp(output_start);end=pd.Timestamp(through) if through is not None else panel.date.max()
    cal=calendar.date
    target=tuple(cal.loc[cal.between(start,end)])
    if not target:raise Blocked('OUTPUT_CALENDAR_EMPTY')
    first=cal.tolist().index(target[0]);lag=lookback['calendar_previous_sessions']
    if first<lag:raise Blocked('ORIGINAL_ARM_CALENDAR_WARMUP_INCOMPLETE')
    required=set(cal.iloc[first-lag:first])|set(target)
    stock_lag=lookback['per_stock_previous_rows']
    for _,g in panel.groupby('code',sort=False):
        g=g.sort_values('date');positions=list(g.index[g.date.between(start,end)])
        if not positions:continue
        first_stock=int(g.index.get_loc(positions[0]))
        required.update(g.date.iloc[max(0,first_stock-stock_lag):first_stock])
        required.update(g.loc[g.date.between(start,end),'date'])
    return tuple(sorted(required))


def _formula_batches(formula,feature_panel,texts,index,*,workers,max_batch_rows):
    """Whole-stock histories only; bounded active work, deterministic reassembly."""
    import pandas as pd
    def work(frame):
        return formula.formula_feature_frame(frame,v193_text=texts['v193'],v192_text=texts['v192'],
            index_close_by_date=index,include_v192_trigger=True,include_v193_trigger=True,include_auditm=True)
    def batches():
        pieces=[];count=0
        for _,g in feature_panel.groupby('code',sort=True):
            if len(g)>max_batch_rows:raise Blocked('WHOLE_STOCK_HISTORY_EXCEEDS_BATCH_BOUND_NO_TRUNCATION')
            if pieces and count+len(g)>max_batch_rows:
                yield pd.concat(pieces,ignore_index=True);pieces=[];count=0
            pieces.append(g);count+=len(g)
        if pieces:yield pd.concat(pieces,ignore_index=True)
    result=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending=[]
        for frame in batches():
            pending.append(pool.submit(work,frame))
            if len(pending)==workers:result.append(pending.pop(0).result())
        result.extend(f.result() for f in pending)
    return pd.concat(result,ignore_index=True).sort_values(['date','code'],kind='mergesort').reset_index(drop=True)


def materialize_candidate(recipe,formula,engine,*,panel,old_panel,old_calendar,index_provider,
                          industry,texts,output_start,output_end,caps,budget):
    """Exact S1 materialize steps, full history retained through original taint."""
    import numpy as np
    import pandas as pd
    spec=recipe.STAGE_SPECS['M0_FORMAL']
    if (tuple(spec['input_arms'])!=tuple(recipe.ARMS) or not spec['requires_common_keys']
            or tuple(recipe.CORE_FEATURES)!=FEATURES):raise Blocked('ORIGINAL_M0_FORMAL_FIVE_ARM_RECIPE_REQUIRED')
    if len(panel)>caps['max_input_rows']:raise Blocked('INPUT_ROW_RESOURCE_BOUND')
    workers=caps['formula_workers']
    if type(workers) is not int or not 1<=workers<=4:raise Blocked('BOUNDED_FORMULA_WORKERS_REQUIRED')
    budget.check('before_calendar')
    cal,calendar_proof=extend_calendar(recipe,panel,old_panel,old_calendar)
    lookback=derive_index_lookback(recipe,formula,engine,texts['v192'],texts['v193'])
    required=required_index_dates(panel,cal,lookback,output_start)
    # Activated loader must call original load_bound_index_series with exactly
    # these dates. No pre-index data are invented for earlier stock warmup.
    index=index_provider(required)
    index=index.copy();index.index=pd.DatetimeIndex(index.index)
    if index.index.duplicated().any() or not set(required).issubset(set(index.index)):
        raise Blocked('TARGET_OR_ACTUAL_INDEX_WARMUP_GAP')
    values=index.loc[list(required)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values<=0).any():raise Blocked('INVALID_REQUIRED_INDEX_VALUES')
    fp,quarantine=recipe.derive_price_feature_view(panel)
    budget.check('before_full_history_formula')
    ff=_formula_batches(formula,fp,texts,index,workers=workers,max_batch_rows=caps['max_formula_batch_rows'])
    if recipe.key_frame_identity(ff)!=recipe.key_frame_identity(fp):raise Blocked('FORMULA_CHANGED_FULL_HISTORY_KEYS')
    budget.check('before_original_five_arms')
    arms=recipe.build_nested_arm_frames(fp,ff,cal,index_close_by_date=index,
        industry_classification=industry,requested_arms=recipe.ARMS)
    ff,arms,taint=recipe.apply_ohlc_taint_masks(ff,arms,cal,quarantine,industry_classification=industry)
    common,columns=recipe.assemble_common_arm_features(arms)
    m0=common['M0']
    if tuple(columns['M0'])!=FEATURES:raise Blocked('ORIGINAL_36_FEATURE_ORDER_CHANGED')
    budget.check('after_original_common_pool')
    early=m0.loc[m0.date.between(pd.Timestamp(output_start),pd.Timestamp(output_end))].reset_index(drop=True)
    return dict(panel=panel,calendar=cal,m0=m0,early_m0=early,
        provenance=dict(calendar=calendar_proof,index_lookback=lookback,index_required_dates_sha256=support.digest(_date_strings(required)),
            index_required_date_count=len(required),index_missing_before_required='RETAINED_UNFILLED_FOR_ORIGINAL_FORMULA',
            warmup_start=panel.date.min().date().isoformat(),output_start=str(output_start),output_end=str(output_end),
            projection_after_formula_arms_taint_and_common=True,input_history_rows=len(panel),early_rows=len(early),
            formula_workers=workers,formula_batch_rows=caps['max_formula_batch_rows'],
            concurrency='WHOLE_STOCK_FORMULA_ONLY; ORIGINAL_CROSS_SECTION_ARMS_AFTER_DETERMINISTIC_JOIN',
            quarantine=quarantine,taint=taint,feature_columns=list(FEATURES),**FLAGS))


def overlap_parity(recipe,candidate,old):
    import pandas as pd
    if tuple(candidate.columns)!=('date','code',*FEATURES) or tuple(old.columns)!=tuple(candidate.columns):
        raise Blocked('EXACT_M0_CACHE_SCHEMA_REQUIRED')
    if old.empty:raise Blocked('OLD_REFERENCE_CACHE_EMPTY')
    new=candidate.loc[candidate.date.between(old.date.min(),old.date.max())].reset_index(drop=True)
    new_keys,old_keys=recipe.key_frame_identity(new),recipe.key_frame_identity(old)
    keys=new_keys==old_keys
    values=(recipe.numeric_matrix_identity(new[list(FEATURES)].to_numpy(),FEATURES)
            ==recipe.numeric_matrix_identity(old[list(FEATURES)].to_numpy(),FEATURES)) if keys else False
    # Do not infer which model is impacted from this global overlap result.
    return dict(status='EXACT' if keys and values else 'DIFFERENT',keys_and_order_exact=keys,
        matrix36_exact=values,old_rows=len(old),candidate_overlap_rows=len(new),
        old_key_identity=old_keys,candidate_key_identity=new_keys)


def classify_model_inputs(recipe,candidate,panel,calendar,bindings,expected_cell_ids):
    import pandas as pd
    if len(set(expected_cell_ids))!=len(expected_cell_ids) or set(bindings)!=set(expected_cell_ids):
        raise Blocked('COMPLETE_UNAMBIGUOUS_ACCEPTED_MODEL_MANIFEST_REQUIRED')
    result={}
    recipe_sha=support.sha(recipe.__file__)
    original_folds={f.fold_id:f for f in recipe.fixed_folds()}
    for cell_id,sidecar in bindings.items():
        identity=sidecar.get('identity',{})
        if (sidecar.get('status')!='COMPLETED' or identity.get('arm')!='M0' or identity.get('horizon') not in (1,2,5)
                or identity.get('code_sha256',{}).get('model_selection')!=recipe_sha
                or tuple(identity.get('data_identity',{}).get('feature_columns',()))!=FEATURES):
            raise Blocked('ACCEPTED_ORIGINAL_M0_MODEL_IDENTITY_REQUIRED')
        fold_id=identity['fold'];year=int(fold_id[:4]);start=pd.Timestamp(year,1 if fold_id[5]=='1' else 7,1)
        end=start+pd.DateOffset(months=6)-pd.Timedelta(days=1)
        if fold_id not in original_folds and fold_id not in ('2022H1','2022H2','2023H1','2023H2'):
            raise Blocked('UNBOUND_EXISTING_MODEL_FOLD')
        fold=original_folds.get(fold_id) or recipe.Fold(fold_id,start,end,False)
        start=fold.start
        h=identity['horizon']
        labels=recipe.build_close_return_labels(panel,horizons=(h,),calendar=calendar,
            signal_start=start-pd.DateOffset(years=2),signal_end_exclusive=start)[h]
        try:
            prepared=recipe.prepare_cell_data(features=candidate,labels=labels,fold=fold,canary=False)
            actual=prepared['identity'];expected=identity['data_identity']
            fields=('training_keys','training_feature_matrix','training_label_vector',
                    'training_sample_weight_vector','test_keys','test_feature_matrix','feature_columns')
            matching={k:actual.get(k)==expected.get(k) for k in fields}
            result[cell_id]=dict(status='UNCHANGED' if all(matching.values()) else 'AFFECTED',matching=matching,
                                actual_identity=actual,expected_identity=expected)
        except recipe.ContractError as exc:
            result[cell_id]=dict(status='AFFECTED_UNMATERIALIZABLE',reason=str(exc))
    return result


def parity_gate(recipe,candidate,old,panel,calendar,bindings,expected_cell_ids):
    overlap=overlap_parity(recipe,candidate,old)
    cells=classify_model_inputs(recipe,candidate,panel,calendar,bindings,expected_cell_ids)
    affected=[k for k,v in cells.items() if v['status']!='UNCHANGED']
    passed=overlap['status']=='EXACT' and not affected
    return dict(status='INPUT_VERSION_PARITY_PASS' if passed else 'INPUT_VERSION_PARITY_FAILED',
        publish_candidate_passport_allowed=passed,old_cache_overlap=overlap,model_cells=cells,
        affected_cell_ids=affected,automatic_refit=False,old_V1_preserved=True,input_versions_mixed=False,
        first_source_pit='CANNOT_CONFIRM',official_calendar='CANNOT_CONFIRM',**FLAGS)


def require_activation(config,activation):
    if config.get('state')!='FROZEN_READY_FOR_ACTIVATION' or config.get('policy')!=POLICY:
        raise Blocked('DRAFT_OR_CHANGED_POLICY_NOT_EXECUTABLE')
    if set(config.get('sources',{}))!={'recipe','formula','engine','fast_builder','s1_runner','v192','v193','loader','support','resource_probe'}:
        raise Blocked('EXACT_ORIGINAL_SOURCE_AND_EXECUTOR_LOADER_BINDINGS_REQUIRED')
    if set(config.get('inputs',{}))!={'prefix','old_qfq','old_calendar','index','industry','old_m0',
                                     'model_manifest','history_proof','accepted_prefix_review','source_plan'}:
        raise Blocked('COMPLETE_ORIGINAL_INPUT_AND_ACCEPTANCE_BINDINGS_REQUIRED')
    for group in ('sources','inputs'):
        if not config.get(group):raise Blocked('COMPLETE_BOUND_DEPENDENCIES_REQUIRED')
        for ref in config[group].values():
            if (not isinstance(ref,dict) or not isinstance(ref.get('path'),str) or not Path(ref['path']).is_absolute()
                    or not isinstance(ref.get('sha256'),str) or len(ref['sha256'])!=64
                    or any(c not in '0123456789abcdef' for c in ref['sha256'])):
                raise Blocked('FINAL_ABSOLUTE_PATH_AND_SHA_REQUIRED')
    if (config.get('accepted_prefix_status')!='INDEPENDENT_ACCEPTED_FULL_HISTORY_CANARY'
            or any(config.get(k)!=v for k,v in FLAGS.items())):raise Blocked('M19_ACCEPTANCE_AND_RESEARCH_BOUNDARY_REQUIRED')
    caps=config['caps']
    for key in ('max_input_rows','max_formula_batch_rows','rss_bytes','min_available_bytes','min_disk_bytes','wall_seconds'):
        if type(caps.get(key)) is not int or caps[key]<=0:raise Blocked('POSITIVE_BOUNDED_ROW_LIMITS_REQUIRED')
    if type(caps.get('formula_workers')) is not int or not 1<=caps['formula_workers']<=4:
        raise Blocked('BOUNDED_FORMULA_WORKERS_REQUIRED')
    if caps.get('threads')!=4:raise Blocked('FOUR_THREAD_BOUNDARY_REQUIRED')
    out=Path(config['output_root'])
    if (not out.is_absolute() or out.parent.resolve()!=BATCH.resolve()
            or not out.name.startswith('M26_') or out.is_symlink()):
        raise Blocked('NEW_OWNED_M26_OUTPUT_REQUIRED')
    if not config.get('expected_cell_ids') or len(set(config['expected_cell_ids']))!=len(config['expected_cell_ids']):
        raise Blocked('FROZEN_COMPLETE_ACCEPTED_CELL_MANIFEST_REQUIRED')
    identity=support.digest({k:config[k] for k in ('policy','sources','inputs','caps','expected_cell_ids')})
    if (activation.get('decision')!='INDEPENDENT_REVIEW_PASS_ACTIVATE'
            or activation.get('scientific_identity')!=identity
            or activation.get('wrapper_sha256')!=support.sha(__file__)
            or not activation.get('review_receipt_sha256') or not activation.get('execution_id')):
        raise Blocked('INDEPENDENT_EXACT_SCOPE_ACTIVATION_REQUIRED')
    return identity


def prepare_registered(config,activation,session,loader):
    """Executor owns original-source IO, official start/terminal and output lock.

    loader.open_once is the bound original-source adapter; it must reuse an
    existing accepted passport/FAST_BIND or narrow changed-input authentication.
    It returns exact original modules and bound frames. No global patch/fold
    mutation or alternate recipe is permitted. Reentry is never automatic.
    """
    identity=require_activation(config,activation)
    with support.deadline(config['caps']['wall_seconds']):
        return _prepare_registered_once(config,activation,session,loader,identity)


def _prepare_registered_once(config,activation,session,loader,identity):
    session.require_active()
    if session.execution_id!=activation['execution_id'] or session.output_exists():
        raise Blocked('REGISTERED_EXECUTION_OR_NEW_OUTPUT_OWNERSHIP_REQUIRED')
    loader_source=Path(inspect.getfile(type(loader))).resolve()
    if (loader_source!=Path(config['sources']['loader']['path']).resolve()
            or support.sha(loader_source)!=config['sources']['loader']['sha256']):
        raise Blocked('REGISTERED_EXECUTOR_LOADER_SOURCE_DRIFT')
    for ref in config['sources'].values():
        if support.sha(ref['path'])!=ref['sha256']:raise Blocked('ORIGINAL_SOURCE_COMPONENT_DRIFT')
    if Path(support.__file__).resolve()!=Path(config['sources']['support']['path']).resolve():
        raise Blocked('SUPPORT_MODULE_IMPORT_COLLISION')
    budget=NativeBudget(config);budget.check('before_bound_input_admission')
    acceptance=support.checked_json(config['inputs']['accepted_prefix_review'])
    if (acceptance.get('schema')!='M26_ORIGINAL_PREFIX_INPUT_ACCEPTANCE_V1'
            or acceptance.get('decision')!='PASS'
            or acceptance.get('prefix_ref')!=config['inputs']['prefix']
            or acceptance.get('history_proof_ref')!=config['inputs']['history_proof']
            or acceptance.get('scope')!='FULL_AVAILABLE_PRE_OLD_HISTORY'):
        raise Blocked('ACTUAL_M19_FULL_HISTORY_ACCEPTANCE_NOT_BOUND')
    loaded=loader.open_once(config,activation,session)
    recipe=loaded['recipe']
    if (loaded.get('admission')!=dict(scientific_identity=identity,bound_inputs=config['inputs'])
            or set(loaded['source_modules'])!={'recipe','formula','engine','fast_builder'}):
        raise Blocked('COMPLETE_ORIGINAL_SOURCE_INPUT_ADMISSION_REQUIRED')
    for name,module in loaded['source_modules'].items():
        ref=config['sources'][name]
        if str(Path(module.__file__).resolve())!=str(Path(ref['path']).resolve()) or support.sha(module.__file__)!=ref['sha256']:
            raise Blocked('ORIGINAL_SOURCE_MODULE_BINDING_DRIFT')
    if any(loaded[k] is not loaded['source_modules'][k] for k in ('recipe','formula','engine')):
        raise Blocked('PRODUCER_OBJECT_DOES_NOT_MATCH_AUTHENTICATED_MODULE')
    for name in ('v192','v193'):
        ref=config['sources'][name]
        if loaded['formula'].exact_formula_source(Path(ref['path']),ref['sha256'])[0]!=loaded['texts'][name]:
            raise Blocked('FORMULA_TEXT_NOT_EXACT_ORIGINAL_SNAPSHOT')
    panel,old,composition=compose_history(recipe,loaded['prefix_raw'],loaded['old_raw'],
        prefix_map=loaded['prefix_map'],old_map=loaded['old_map'],history_proof=loaded['history_proof'],
        max_rows=config['caps']['max_input_rows'])
    candidate=materialize_candidate(recipe,loaded['formula'],loaded['engine'],panel=panel,old_panel=old,
        old_calendar=loaded['old_calendar'],index_provider=loaded['index_provider'],industry=loaded['industry'],
        texts=loaded['texts'],output_start=POLICY['output_start'],output_end=POLICY['output_end'],
        caps=config['caps'],budget=budget)
    result=parity_gate(recipe,candidate['m0'],loaded['old_m0'],panel,candidate['calendar'],
                       loaded['model_bindings'],config['expected_cell_ids'])
    budget.check('after_model_input_parity')
    return dict(scientific_identity=identity,composition=composition,candidate=candidate,parity=result,
                actual_execution_id=session.execution_id,base_fits=0,performance_evaluation=False,**FLAGS)
