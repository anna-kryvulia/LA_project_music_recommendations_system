def choose_best_k(recommender_class,train_matrix,test_df,user_to_index,artist_to_index,index_to_artist,evaluate_function,k_values=None,top_n=10,tolerance=0.002):
    ''' 
    Evaluates the recommender model with different n_factors (k) and selects the best one based on Precision@k.
    '''
    if k_values is None:
        k_values = [5, 10, 15, 20, 30, 40, 50, 75, 100]

    results = []

    for n_factors in k_values:
        print(f"\nTesting n_factors = {n_factors}")

        model = recommender_class(n_factors=n_factors)
        model.fit(train_matrix)

        predicted_matrix = model.reconstruct_matrix()

        metrics = evaluate_function(
            train_matrix=train_matrix,
            predicted_matrix=predicted_matrix,
            test_df=test_df,
            user_to_index=user_to_index,
            artist_to_index=artist_to_index,
            index_to_artist=index_to_artist,
            k=top_n
        )

        result = {
            "n_factors": n_factors,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "hit_rate": metrics["hit_rate"]
        }

        results.append(result)

        print(
            f"n_factors={n_factors} | "
            f"Precision@{top_n}: {metrics['precision']:.4f} | "
            f"Recall@{top_n}: {metrics['recall']:.4f} | "
            f"HitRate@{top_n}: {metrics['hit_rate']:.4f}"
        )

    best_precision = max(result["precision"] for result in results)

    good_candidates = [
        result for result in results
        if result["precision"] >= best_precision - tolerance
    ]

    best_result = min(good_candidates, key=lambda result: result["n_factors"])
    best_k = best_result["n_factors"]

    print("\nBEST K")
    print("======")
    print(f"Best n_factors: {best_k}")
    print(f"Precision:      {best_result['precision']:.4f}")
    print(f"Recall:         {best_result['recall']:.4f}")
    print(f"HitRate:        {best_result['hit_rate']:.4f}")

    return best_k, results