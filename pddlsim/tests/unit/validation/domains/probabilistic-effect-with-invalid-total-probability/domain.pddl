(define (domain probabilistic-effect-without-requirement)
        (:requirements :probabilistic-effects)
        (:action move
         :parameters ()
         :effect (probabilistic 0.6 (and) 0.6 (and))))