# Exact nonnegative sums retain the weight exponent until after multiplication
# and accumulation. A zero Float64 display value can still have a finite log sum.
_dmmm_lse(x::Real) = x
_dmmm_lse(x::Real,y::Real,z::Real...) = _dmmm_lse(_dmmm_logaddexp(x,y),z...)

function _dmmm_logweight_sum(logweights, values=nothing; rows=eachindex(logweights))
    result=-Inf
    for i in rows
        w=logweights[i]
        w==-Inf && continue
        if isnothing(values)
            term=w
        else
            x=values[i]
            isfinite(x) && x>=0 || throw(ArgumentError("Expected finite nonnegative sufficient-statistic factors."))
            x==0 && continue
            term=w+log(x)
        end
        result=_dmmm_logaddexp(result,term)
    end
    return result
end

function _dmmm_signed_weighted_sum(logweights, values)
    positive=negative=-Inf
    for i in eachindex(logweights,values)
        w,x=logweights[i],values[i]
        (w==-Inf || x==0) && continue
        isfinite(x) || throw(ArgumentError("Nonfinite signed sufficient-statistic factor."))
        term=w+log(abs(x))
        x>0 ? (positive=_dmmm_logaddexp(positive,term)) :
              (negative=_dmmm_logaddexp(negative,term))
    end
    positive==negative && return 0.0
    # exp(max)*expm1(diff) avoids cancellation between nearly equal signed sums,
    # with the final exponent delayed so representable small results survive.
    hi,lo=max(positive,negative),min(positive,negative)
    value=exp(hi+log(-expm1(lo-hi)))
    return positive>negative ? value : -value
end

function _dmmm_responsibility_logs(responsibilities, supplied=nothing)
    if isnothing(supplied)
        all(x->isfinite(x)&&x>=0,responsibilities) || throw(ArgumentError("Invalid responsibilities."))
        return log.(responsibilities)
    end
    size(supplied)==size(responsibilities) || throw(DimensionMismatch("Responsibility log dimensions differ."))
    all(x->isfinite(x)||x==-Inf,supplied) || throw(ArgumentError("Invalid responsibility logs."))
    return supplied
end

"""Exact O(NK) E-step for T<=3, using positive log-polynomials throughout.

For each nonempty subset A of periods, m_A=sum_k alpha_k*prod_{t in A} e_tk.
The Dirichlet moment numerator is m_1*m_2*m_3+m_12*m_3+m_13*m_2+
m_23*m_1+2*m_123 (with its T=1/2 restrictions). Prefix/suffix log sums
give leave-one-class-out moments without subtraction. Marginalize complete
class-count patterns before exponentiating, retaining log responsibilities for
downstream sufficient statistics even below Float64 probability range.
"""
function dmmm_estep_closed_form(expert_loglik,X,group,α,ϕ)
    v=_dmmm_validate_estep_inputs(expert_loglik,X,group,α,ϕ)
    maximum(length,v.group_index)<=3 || throw(ArgumentError("Closed form supports at most three periods."))
    K=v.n_components
    logr=fill(-Inf,v.n_observations,K);xi=zeros(length(v.group_ids),K)
    gl=zeros(length(v.group_ids))
    ranges=_dmmm_policy_ranges(length(v.group_index))
    scratch=[(
        e=zeros(3,K),la=zeros(K),prefix=fill(-Inf,7,K+1),
        suffix=fill(-Inf,7,K+1),mom=zeros(7),x=zeros(7),shifts=zeros(3),
    ) for _ in ranges]
    @threads :static for chunk in eachindex(ranges)
        e=scratch[chunk].e;la=scratch[chunk].la
        prefix=scratch[chunk].prefix;suffix=scratch[chunk].suffix
        mom=scratch[chunk].mom;x=scratch[chunk].x;shifts=scratch[chunk].shifts
        for i in ranges[chunk]
            rows=v.group_index[i];T=length(rows);masks=(1<<T)-1
            for k in 1:K;la[k]=log(ϕ)+log(v.gate_prob[i,k]);end
            for t in 1:T
                shifts[t]=maximum(view(expert_loglik,rows[t],:))
                isfinite(shifts[t]) || throw(ArgumentError("A period has zero likelihood under every component."))
                for k in 1:K;e[t,k]=expert_loglik[rows[t],k]-shifts[t];end
            end
            for mask in 1:masks
                prefix[mask,1]=-Inf;suffix[mask,K+1]=-Inf
                for k in 1:K
                    term=la[k]
                    for t in 1:T;(mask & (1<<(t-1)))!=0 && (term+=e[t,k]);end
                    prefix[mask,k+1]=_dmmm_logaddexp(prefix[mask,k],term)
                end
                for k in K:-1:1
                    term=la[k]
                    for t in 1:T;(mask & (1<<(t-1)))!=0 && (term+=e[t,k]);end
                    suffix[mask,k]=_dmmm_logaddexp(suffix[mask,k+1],term)
                end
                mom[mask]=prefix[mask,K+1]
            end
            z=T==1 ? mom[1] : T==2 ? _dmmm_lse(mom[1]+mom[2],mom[3]) :
                _dmmm_lse(mom[1]+mom[2]+mom[4],mom[3]+mom[4],mom[5]+mom[2],mom[6]+mom[1],log(2.)+mom[7])
            isfinite(z) || throw(ArgumentError("Every class path has zero/nonfinite likelihood."))
            gl[i]=sum(view(shifts,1:T))+z-sum(log(ϕ+t) for t in 0:T-1)
            psi_total=digamma(ϕ+T)
            for k in 1:K
                for mask in 1:masks;x[mask]=_dmmm_logaddexp(prefix[mask,k],suffix[mask,k+1]);end
                a=ϕ*v.gate_prob[i,k]
                a>0 || throw(ArgumentError("Dirichlet parameter is outside representable positive range."))
                l1=la[k];l2=l1+log1p(a);l3=l2+log(a+2.)
                e1=e[1,k]
                if T==1
                    w0=x[1];w1=l1+e1;w2=w3=-Inf
                    logr[rows[1],k]=w1-z
                elseif T==2
                    e2=e[2,k]
                    w10=l1+e1+x[2];w01=l1+e2+x[1];w2=l2+e1+e2
                    w0=_dmmm_lse(x[1]+x[2],x[3]);w1=_dmmm_lse(w10,w01);w3=-Inf
                    logr[rows[1],k]=_dmmm_lse(w10,w2)-z
                    logr[rows[2],k]=_dmmm_lse(w01,w2)-z
                else
                    e2,e3=e[2,k],e[3,k]
                    w100=l1+e1+_dmmm_lse(x[2]+x[4],x[6])
                    w010=l1+e2+_dmmm_lse(x[1]+x[4],x[5])
                    w001=l1+e3+_dmmm_lse(x[1]+x[2],x[3])
                    w110=l2+e1+e2+x[4];w101=l2+e1+e3+x[2];w011=l2+e2+e3+x[1]
                    w3=l3+e1+e2+e3
                    w0=_dmmm_lse(x[1]+x[2]+x[4],x[3]+x[4],x[5]+x[2],x[6]+x[1],log(2.)+x[7])
                    w1=_dmmm_lse(w100,w010,w001);w2=_dmmm_lse(w110,w101,w011)
                    logr[rows[1],k]=_dmmm_lse(w100,w110,w101,w3)-z
                    logr[rows[2],k]=_dmmm_lse(w010,w110,w011,w3)-z
                    logr[rows[3],k]=_dmmm_lse(w001,w101,w011,w3)-z
                end
                logabsxi=-Inf
                for (c,w) in enumerate((w0,w1,w2,w3))
                    w==-Inf && continue
                    difference=digamma(a+(c-1))-psi_total
                    difference<=0 || error("Invalid positive expected-log-membership term.")
                    difference==0 && continue
                    logabsxi=_dmmm_logaddexp(logabsxi,w-z+log(-difference))
                end
                xi[i,k]=-exp(logabsxi)
            end
        end
    end
    return _dmmm_finish_estep(:closed_form,(special_case=:T_le_3,complexity=:O_NK,numerics=:log_patterns),
        sum(gl),gl,exp.(logr),xi,v,ϕ;log_responsibilities=logr)
end
