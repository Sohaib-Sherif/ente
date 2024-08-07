package filedata

import (
	"context"
	"database/sql"
	"fmt"
	"github.com/ente-io/museum/ente"
	"github.com/ente-io/museum/ente/filedata"
	"github.com/ente-io/stacktrace"
	"github.com/lib/pq"
)

// Repository defines the methods for inserting, updating, and retrieving file data.
type Repository struct {
	DB *sql.DB
}

const (
	ReplicationColumn = "replicated_buckets"
	DeletionColumn    = "delete_from_buckets"
	InflightRepColumn = "inflight_rep_buckets"
)

func (r *Repository) InsertOrUpdate(ctx context.Context, data filedata.Row) error {
	query := `
        INSERT INTO file_data 
            (file_id, user_id, data_type, size, latest_bucket) 
        VALUES 
            ($1, $2, $3, $4, $5)
        ON CONFLICT (file_id, data_type)
        DO UPDATE SET 
            size = EXCLUDED.size,
            delete_from_buckets = array(
                SELECT DISTINCT elem FROM unnest(
                    array_append(
                        array_cat(array_cat(file_data.replicated_buckets, file_data.delete_from_buckets), file_data.inflight_rep_buckets),
                        CASE WHEN file_data.latest_bucket != EXCLUDED.latest_bucket THEN file_data.latest_bucket  END
                    )
                ) AS elem
                WHERE elem IS NOT NULL AND elem != EXCLUDED.latest_bucket
            ),
            replicated_buckets = ARRAY[]::s3region[],
            latest_bucket = EXCLUDED.latest_bucket,
            updated_at = now_utc_micro_seconds()
        WHERE file_data.is_deleted = false`
	_, err := r.DB.ExecContext(ctx, query,
		data.FileID, data.UserID, string(data.Type), data.Size, data.LatestBucket)
	if err != nil {
		return stacktrace.Propagate(err, "failed to insert file data")
	}
	return nil
}

func (r *Repository) GetFilesData(ctx context.Context, oType ente.ObjectType, fileIDs []int64) ([]filedata.Row, error) {
	rows, err := r.DB.QueryContext(ctx, `SELECT file_id, user_id, data_type, size, latest_bucket, replicated_buckets, delete_from_buckets, inflight_rep_buckets, pending_sync, is_deleted, last_sync_time, created_at, updated_at
										FROM file_data
										WHERE data_type = $1 AND file_id = ANY($2)`, string(oType), pq.Array(fileIDs))
	if err != nil {
		return nil, stacktrace.Propagate(err, "")
	}
	return convertRowsToFilesData(rows)
}

func (r *Repository) GetFileData(ctx context.Context, fileIDs int64) ([]filedata.Row, error) {
	rows, err := r.DB.QueryContext(ctx, `SELECT file_id, user_id, data_type, size, latest_bucket, replicated_buckets, delete_from_buckets,inflight_rep_buckets, pending_sync, is_deleted, last_sync_time, created_at, updated_at
										FROM file_data
										WHERE file_id = $1`, fileIDs)
	if err != nil {
		return nil, stacktrace.Propagate(err, "")
	}
	return convertRowsToFilesData(rows)
}

func (r *Repository) AddBucket(row filedata.Row, bucketID string, columnName string) error {
	query := fmt.Sprintf(`
        UPDATE file_data
        SET %s = array(
            SELECT DISTINCT elem FROM unnest(
                array_append(file_data.%s, $1)
            ) AS elem
        )
        WHERE file_id = $2 AND data_type = $3 and user_id = $4`, columnName, columnName)
	result, err := r.DB.Exec(query, bucketID, row.FileID, string(row.Type), row.UserID)
	if err != nil {
		return stacktrace.Propagate(err, "failed to add bucket to "+columnName)
	}
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return stacktrace.Propagate(err, "")
	}
	if rowsAffected == 0 {
		return stacktrace.NewError("bucket not added to " + columnName)
	}
	return nil
}

func (r *Repository) RemoveBucket(row filedata.Row, bucketID string, columnName string) error {
	query := fmt.Sprintf(`
  UPDATE file_data
  SET %s = array(
   SELECT DISTINCT elem FROM unnest(
    array_remove(
     file_data.%s,
     $1
    )
   ) AS elem
   WHERE elem IS NOT NULL
  )
  WHERE file_id = $2 AND data_type = $3 and user_id = $4`, columnName, columnName)
	result, err := r.DB.Exec(query, bucketID, row.FileID, string(row.Type), row.UserID)
	if err != nil {
		return stacktrace.Propagate(err, "failed to remove bucket from "+columnName)
	}
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return stacktrace.Propagate(err, "")
	}
	if rowsAffected == 0 {
		return stacktrace.NewError("bucket not removed from " + columnName)
	}
	return nil
}

func (r *Repository) MoveBetweenBuckets(row filedata.Row, bucketID string, sourceColumn string, destColumn string) error {
	query := fmt.Sprintf(`
  UPDATE file_data
  SET %s = array(
   SELECT DISTINCT elem FROM unnest(
    array_append(
     file_data.%s,
     $1
    )
   ) AS elem
   WHERE elem IS NOT NULL
  ),
  %s = array(
   SELECT DISTINCT elem FROM unnest(
    array_remove(
     file_data.%s,
     $1
    )
   ) AS elem
   WHERE elem IS NOT NULL
  )
  WHERE file_id = $2 AND data_type = $3 and user_id = $4`, destColumn, destColumn, sourceColumn, sourceColumn)
	result, err := r.DB.Exec(query, bucketID, row.FileID, string(row.Type), row.UserID)
	if err != nil {
		return stacktrace.Propagate(err, "failed to move bucket from "+sourceColumn+" to "+destColumn)
	}
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return stacktrace.Propagate(err, "")
	}
	if rowsAffected == 0 {
		return stacktrace.NewError("bucket not moved from " + sourceColumn + " to " + destColumn)
	}
	return nil
}

func (r *Repository) DeleteFileData(ctx context.Context, row filedata.Row) error {
	query := `
DELETE FROM file_data
WHERE file_id = $1 AND data_type = $2 AND latest_bucket = $3 AND user_id = $4 AND replicated_buckets = ARRAY[]::s3region[] AND delete_from_buckets = ARRAY[]::s3region[]`
	res, err := r.DB.ExecContext(ctx, query, row.FileID, string(row.Type), row.LatestBucket, row.UserID)
	if err != nil {
		return stacktrace.Propagate(err, "")
	}
	rowsAffected, err := res.RowsAffected()
	if err != nil {
		return stacktrace.Propagate(err, "")
	}
	if rowsAffected == 0 {
		return stacktrace.NewError("file data not deleted")
	}
	return nil

}

func convertRowsToFilesData(rows *sql.Rows) ([]filedata.Row, error) {
	var filesData []filedata.Row
	for rows.Next() {
		var fileData filedata.Row
		err := rows.Scan(&fileData.FileID, &fileData.UserID, &fileData.Type, &fileData.Size, &fileData.LatestBucket, pq.Array(&fileData.ReplicatedBuckets), pq.Array(&fileData.DeleteFromBuckets), pq.Array(&fileData.InflightReplicas), &fileData.PendingSync, &fileData.IsDeleted, &fileData.LastSyncTime, &fileData.CreatedAt, &fileData.UpdatedAt)
		if err != nil {
			return nil, stacktrace.Propagate(err, "")
		}
		filesData = append(filesData, fileData)
	}
	return filesData, nil
}
