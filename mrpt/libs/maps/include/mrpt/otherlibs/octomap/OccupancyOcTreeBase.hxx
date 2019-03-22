// $Id: OccupancyOcTreeBase.hxx 440 2012-11-02 10:41:10Z ahornung $

/**
 * OctoMap:
 * A probabilistic, flexible, and compact 3D mapping library for robotic systems.
 * @author K. M. Wurm, A. Hornung, University of Freiburg, Copyright (C) 2009.
 * @see http://octomap.sourceforge.net/
 * License: New BSD License
 */

/*
 * Copyright (c) 2009-2011, K. M. Wurm, A. Hornung, University of Freiburg
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University of Freiburg nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <bitset>

namespace octomap {

  template <class NODE>
  OccupancyOcTreeBase<NODE>::OccupancyOcTreeBase(double resolution)
    : OcTreeBaseImpl<NODE,AbstractOccupancyOcTree>(resolution), use_bbx_limit(false), use_change_detection(false)
  {

  }

  template <class NODE>
  OccupancyOcTreeBase<NODE>::OccupancyOcTreeBase(double resolution, unsigned int tree_depth, unsigned int tree_max_val)
    : OcTreeBaseImpl<NODE,AbstractOccupancyOcTree>(resolution, tree_depth, tree_max_val), use_bbx_limit(false), use_change_detection(false)
  {

  }

  template <class NODE>
  OccupancyOcTreeBase<NODE>::~OccupancyOcTreeBase(){
  }

  template <class NODE>
  OccupancyOcTreeBase<NODE>::OccupancyOcTreeBase(const OccupancyOcTreeBase<NODE>& rhs) :
  OcTreeBaseImpl<NODE,AbstractOccupancyOcTree>(rhs), use_bbx_limit(rhs.use_bbx_limit),
    bbx_min(rhs.bbx_min), bbx_max(rhs.bbx_max),
    bbx_min_key(rhs.bbx_min_key), bbx_max_key(rhs.bbx_max_key),
    use_change_detection(rhs.use_change_detection), changed_keys(rhs.changed_keys)
  {
    this->clamping_thres_min = rhs.clamping_thres_min;
    this->clamping_thres_max = rhs.clamping_thres_max;
    this->prob_hit_log = rhs.prob_hit_log;
    this->prob_miss_log = rhs.prob_miss_log;
    this->occ_prob_thres_log = rhs.occ_prob_thres_log;


  }

  // performs transformation to data and sensor origin first
  template <class NODE>
  void OccupancyOcTreeBase<NODE>::insertScan(const ScanNode& scan, double maxrange, bool pruning, bool lazy_eval) {
    Pointcloud& cloud = *(scan.scan);
    pose6d frame_origin = scan.pose;
    point3d sensor_origin = frame_origin.inv().transform(scan.pose.trans());
    insertScan(cloud, sensor_origin, frame_origin, maxrange, pruning, lazy_eval);
  }


  template <class NODE>
  void OccupancyOcTreeBase<NODE>::insertScan(const Pointcloud& scan, const octomap::point3d& sensor_origin,
                                             double maxrange, bool pruning, bool lazy_eval) {

    KeySet free_cells, occupied_cells;
    computeUpdate(scan, sensor_origin, free_cells, occupied_cells, maxrange);

    // insert data into tree  -----------------------
    for (KeySet::iterator it = free_cells.begin(); it != free_cells.end(); ++it) {
      updateNode(*it, false, lazy_eval);
    }
    for (KeySet::iterator it = occupied_cells.begin(); it != occupied_cells.end(); ++it) {
      updateNode(*it, true, lazy_eval);
    }

    // TODO: does pruning make sense if we used "lazy_eval"?
    if (pruning) this->prune();
  }

  template <class NODE>
  template <class SCAN>
  void OccupancyOcTreeBase<NODE>::insertScan_templ(const SCAN& scan, const octomap::point3d& sensor_origin,
                                             double maxrange, bool pruning, bool lazy_eval) {

    KeySet free_cells, occupied_cells;
    computeUpdate(scan, sensor_origin, free_cells, occupied_cells, maxrange);

    // insert data into tree  -----------------------
    for (KeySet::iterator it = free_cells.begin(); it != free_cells.end(); ++it) {
      updateNode(*it, false, lazy_eval);
    }
    for (KeySet::iterator it = occupied_cells.begin(); it != occupied_cells.end(); ++it) {
      updateNode(*it, true, lazy_eval);
    }

    // TODO: does pruning make sense if we used "lazy_eval"?
    if (pruning) this->prune();
  }

  // performs transformation to data and sensor origin first
  template <class NODE>
  void OccupancyOcTreeBase<NODE>::insertScan(const Pointcloud& pc, const point3d& sensor_origin, const pose6d& frame_origin,
                                             double maxrange, bool pruning, bool lazy_eval) {
    Pointcloud transformed_scan (pc);
    transformed_scan.transform(frame_origin);
    point3d transformed_sensor_origin = frame_origin.transform(sensor_origin);
    insertScan(transformed_scan, transformed_sensor_origin, maxrange, pruning, lazy_eval);
  }


  template <class NODE>
  void OccupancyOcTreeBase<NODE>::insertScanNaive(const Pointcloud& pc, const point3d& origin, double maxrange, bool pruning, bool lazy_eval) {
    if (pc.size() < 1)
      return;

    // integrate each single beam
    octomap::point3d p;
    for (octomap::Pointcloud::const_iterator point_it = pc.begin();
         point_it != pc.end(); ++point_it) {
      this->insertRay(origin, *point_it, maxrange, lazy_eval);
    }

    if (pruning)
      this->prune();
  }



  template <class NODE>
  inline void OccupancyOcTreeBase<NODE>::computeUpdate_onePoint(const point3d& p, const octomap::point3d& origin,
                    KeySet& free_cells, KeySet& occupied_cells, double maxrange)
  {
    if (!use_bbx_limit) {
    // -------------- no BBX specified ---------------
    if ((maxrange < 0.0) || ((p - origin).norm() <= maxrange) ) { // is not maxrange meas.
        // free cells
        if (this->computeRayKeys(origin, p, this->keyray)){
        free_cells.insert(this->keyray.begin(), this->keyray.end());
        }
        // occupied endpoint
        OcTreeKey key;
        if (this->coordToKeyChecked(p, key))
        occupied_cells.insert(key);
    } // end if NOT maxrange

    else { // user set a maxrange and this is reached
        point3d direction = (p - origin).normalized ();
        point3d new_end = origin + direction * (float) maxrange;
        if (this->computeRayKeys(origin, new_end, this->keyray)){
        free_cells.insert(this->keyray.begin(), this->keyray.end());
        }
    } // end if maxrange
    }
    // --- update limited by user specified BBX  -----
    else {
    // endpoint in bbx and not maxrange?
    if ( inBBX(p) && ((maxrange < 0.0) || ((p - origin).norm () <= maxrange) ) )  {

        // occupied endpoint
        OcTreeKey key;
        if (this->coordToKeyChecked(p, key))
        occupied_cells.insert(key);

        // update freespace, break as soon as bbx limit is reached
        if (this->computeRayKeys(origin, p, this->keyray)){
        for(KeyRay::reverse_iterator rit=this->keyray.rbegin(); rit != this->keyray.rend(); ++rit) {
            if (inBBX(*rit)) {
            free_cells.insert(*rit);
            }
            else break;
        }
        } // end if compute ray
    } // end if in BBX and not maxrange
    } // end bbx case
  }

	template <class NODE>
	template <class POINTCLOUD>
	void OccupancyOcTreeBase<NODE>::computeUpdate_templ( const POINTCLOUD& scan, const octomap::point3d& origin, KeySet& free_cells, KeySet& occupied_cells, double maxrange)
	{
		const size_t N = scan.size();

		const std::vector<float> &xs = scan.getPointsBufferRef_x();
		const std::vector<float> &ys = scan.getPointsBufferRef_y();
		const std::vector<float> &zs = scan.getPointsBufferRef_z();

		for (size_t i=0;i<N;i++)
		{
			const point3d p(xs[i],ys[i],zs[i]);
			computeUpdate_onePoint(p,origin,free_cells,occupied_cells,maxrange);
		}
		// prefer occupied cells over free ones (and make sets disjunct)
		for(KeySet::iterator it = free_cells.begin(), end=free_cells.end(); it!= end; )
		{
			if (occupied_cells.find(*it) != occupied_cells.end())
			     it = free_cells.erase(it);
			else ++it;
		}
	}

  template <class NODE>
  void OccupancyOcTreeBase<NODE>::computeUpdate(const Pointcloud& scan, const octomap::point3d& origin,
                                                KeySet& free_cells, KeySet& occupied_cells,
                                                double maxrange)
  {
    //#pragma omp parallel private (local_key_ray, point_it)
    for (Pointcloud::const_iterator point_it = scan.begin(); point_it != scan.end(); ++point_it) {
      const point3d& p = *point_it;
	  computeUpdate_onePoint(p,origin,free_cells,occupied_cells,maxrange);
    }
	// prefer occupied cells over free ones (and make sets disjunct)
	for(KeySet::iterator it = free_cells.begin(), end=free_cells.end(); it!= end; ){
		if (occupied_cells.find(*it) != occupied_cells.end()){
		it = free_cells.erase(it);
		} else {
		++it;
		}
	}
  }

  template <class NODE>
  NODE* OccupancyOcTreeBase<NODE>::updateNode(const OcTreeKey& key, float log_odds_update, bool lazy_eval) {
    return updateNodeRecurs(this->root, false, key, 0, log_odds_update, lazy_eval);
  }

  template <class NODE>
  NODE* OccupancyOcTreeBase<NODE>::updateNode(const point3d& value, float log_odds_update, bool lazy_eval) {
    OcTreeKey key;
    if (!this->coordToKeyChecked(value, key))
      return NULL;
    return updateNode(key, log_odds_update, lazy_eval);
  }

  template <class NODE>
  NODE* OccupancyOcTreeBase<NODE>::updateNode(double x, double y, double z, float log_odds_update, bool lazy_eval) {
    OcTreeKey key;
    if (!this->coordToKeyChecked(x, y, z, key))
      return NULL;
    return updateNode(key, log_odds_update, lazy_eval);
  }

  template <class NODE>
  NODE* OccupancyOcTreeBase<NODE>::updateNode(const OcTreeKey& key, bool occupied, bool lazy_eval) {
    NODE* leaf = this->search(key);
    // no change: node already at threshold
    if (leaf && (this->isNodeAtThreshold(leaf)) && (this->isNodeOccupied(leaf) == occupied)) {
      return leaf;
    }
    if (occupied) return updateNodeRecurs(this->root, false, key, 0, this->prob_hit_log,  lazy_eval);
    else          return updateNodeRecurs(this->root, false, key, 0, this->prob_miss_log, lazy_eval);
  }

  template <class NODE>
  NODE* OccupancyOcTreeBase<NODE>::updateNode(const point3d& value, bool occupied, bool lazy_eval) {
    OcTreeKey key;
    if (!this->coordToKeyChecked(value, key))
      return NULL;
    return updateNode(key, occupied, lazy_eval);
  }

  template <class NODE>
  NODE* OccupancyOcTreeBase<NODE>::updateNode(double x, double y, double z, bool occupied, bool lazy_eval) {
    OcTreeKey key;
    if (!this->coordToKeyChecked(x, y, z, key))
      return NULL;
    return updateNode(key, occupied, lazy_eval);
  }

  template <class NODE>
  NODE* OccupancyOcTreeBase<NODE>::updateNodeRecurs(NODE* node, bool node_just_created, const OcTreeKey& key,
                                                    unsigned int depth, const float& log_odds_update, bool lazy_eval) {
    unsigned int pos = computeChildIdx(key, this->tree_depth-1-depth);
    bool created_node = false;

    // follow down to last level
    if (depth < this->tree_depth) {
      if (!node->childExists(pos)) {
        // child does not exist, but maybe it's a pruned node?
        if ((!node->hasChildren()) && !node_just_created && (node != this->root)) {
          // current node does not have children AND it is not a new node
          // AND its not the root node
          // -> expand pruned node
          node->expandNode();
          this->tree_size+=8;
          this->size_changed = true;
        }
        else {
          // not a pruned node, create requested child
          node->createChild(pos);
          this->tree_size++;
          this->size_changed = true;
          created_node = true;
        }
      }

      if (lazy_eval)
        return updateNodeRecurs(node->getChild(pos), created_node, key, depth+1, log_odds_update, lazy_eval);
      else {
        NODE* retval = updateNodeRecurs(node->getChild(pos), created_node, key, depth+1, log_odds_update, lazy_eval);
        // set own probability according to prob of children
        node->updateOccupancyChildren();
        return retval;
      }
    }

    // at last level, update node, end of recursion
    else {
      if (use_change_detection) {
        bool occBefore = this->isNodeOccupied(node);
        updateNodeLogOdds(node, log_odds_update);
        if (node_just_created){  // new node
          changed_keys.insert(std::pair<OcTreeKey,bool>(key, true));
        } else if (occBefore != this->isNodeOccupied(node)) {  // occupancy changed, track it
          KeyBoolMap::iterator it = changed_keys.find(key);
          if (it == changed_keys.end())
            changed_keys.insert(std::pair<OcTreeKey,bool>(key, false));
          else if (it->second == false)
            changed_keys.erase(it);
        }
      }
      else {
        updateNodeLogOdds(node, log_odds_update);
      }
      return node;
    }
  }


  template <class NODE>
  void OccupancyOcTreeBase<NODE>::calcNumThresholdedNodes(unsigned int& num_thresholded,
                                       unsigned int& num_other) const {
    num_thresholded = 0;
    num_other = 0;
    // TODO: The recursive call could be completely replaced with the new iterators
    calcNumThresholdedNodesRecurs(this->root, num_thresholded, num_other);
  }

  template <class NODE>
  void OccupancyOcTreeBase<NODE>::calcNumThresholdedNodesRecurs (NODE* node,
                                              unsigned int& num_thresholded,
                                              unsigned int& num_other) const {
    assert(node != NULL);
    for (unsigned int i=0; i<8; i++) {
      if (node->childExists(i)) {
        NODE* child_node = node->getChild(i);
        if (this->isNodeAtThreshold(child_node))
          num_thresholded++;
        else
          num_other++;
        calcNumThresholdedNodesRecurs(child_node, num_thresholded, num_other);
      } // end if child
    } // end for children
  }

  template <class NODE>
  void OccupancyOcTreeBase<NODE>::updateInnerOccupancy(){
    this->updateInnerOccupancyRecurs(this->root, 0);
  }

  template <class NODE>
  void OccupancyOcTreeBase<NODE>::updateInnerOccupancyRecurs(NODE* node, unsigned int depth){
    // only recurse and update for inner nodes:
    if (node->hasChildren()){
      // return early for last level:
      if (depth < this->tree_depth){
        for (unsigned int i=0; i<8; i++) {
          if (node->childExists(i)) {
            updateInnerOccupancyRecurs(node->getChild(i), depth+1);
          }
        }
      }
      node->updateOccupancyChildren();
    }
  }

  template <class NODE>
  void OccupancyOcTreeBase<NODE>::toMaxLikelihood() {

    // convert bottom up
    for (unsigned int depth=this->tree_depth; depth>0; depth--) {
      toMaxLikelihoodRecurs(this->root, 0, depth);
    }

    // convert root
    nodeToMaxLikelihood(this->root);
  }

  template <class NODE>
  void OccupancyOcTreeBase<NODE>::toMaxLikelihoodRecurs(NODE* node, unsigned int depth,
      unsigned int max_depth) {

    if (depth < max_depth) {
      for (unsigned int i=0; i<8; i++) {
        if (node->childExists(i)) {
          toMaxLikelihoodRecurs(node->getChild(i), depth+1, max_depth);
        }
      }
    }

    else { // max level reached
      nodeToMaxLikelihood(node);
    }
  }

  template <class NODE>
  bool OccupancyOcTreeBase<NODE>::castRay(const point3d& origin, const point3d& directionP, point3d& end,
                                          bool ignoreUnknown, double maxRange) const {

    /// ----------  see OcTreeBase::computeRayKeys  -----------

    // Initialization phase -------------------------------------------------------
    OcTreeKey current_key;
    if ( !OcTreeBaseImpl<NODE,AbstractOccupancyOcTree>::coordToKeyChecked(origin, current_key) ) {
      OCTOMAP_WARNING_STR("Coordinates out of bounds during ray casting");
      return false;
    }

    NODE* startingNode = this->search(current_key);
    if (startingNode){
      if (this->isNodeOccupied(startingNode)){
        // Occupied node found at origin
        // (need to convert from key, since origin does not need to be a voxel center)
        end = this->keyToCoord(current_key);
        return true;
      }
    } else if(!ignoreUnknown){
      OCTOMAP_ERROR_STR("Origin node at " << origin << " for raycasting not found, does the node exist?");
      end = this->keyToCoord(current_key);
      return false;
    }

    point3d direction = directionP.normalized();
    bool max_range_set = (maxRange > 0.0);

    int step[3];
    double tMax[3];
    double tDelta[3];

    for(unsigned int i=0; i < 3; ++i) {
      // compute step direction
      if (direction(i) > 0.0) step[i] =  1;
      else if (direction(i) < 0.0)   step[i] = -1;
      else step[i] = 0;

      // compute tMax, tDelta
      if (step[i] != 0) {
        // corner point of voxel (in direction of ray)
        double voxelBorder = this->keyToCoord(current_key[i]);
        voxelBorder += double(step[i] * this->resolution * 0.5);

        tMax[i] = ( voxelBorder - origin(i) ) / direction(i);
        tDelta[i] = this->resolution / fabs( direction(i) );
      }
      else {
        tMax[i] =  std::numeric_limits<double>::max();
        tDelta[i] = std::numeric_limits<double>::max();
      }
    }

    if (step[0] == 0 && step[1] == 0 && step[2] == 0){
    	OCTOMAP_ERROR("Raycasting in direction (0,0,0) is not possible!");
    	return false;
    }

    // for speedup:
    double maxrange_sq = maxRange *maxRange;

    // Incremental phase  ---------------------------------------------------------

    bool done = false;

    while (!done) {
      unsigned int dim;

      // find minimum tMax:
      if (tMax[0] < tMax[1]){
        if (tMax[0] < tMax[2]) dim = 0;
        else                   dim = 2;
      }
      else {
        if (tMax[1] < tMax[2]) dim = 1;
        else                   dim = 2;
      }

      // check for overflow:
      if ((step[dim] < 0 && current_key[dim] == 0)
    		  || (step[dim] > 0 && current_key[dim] == 2* this->tree_max_val-1))
      {
        OCTOMAP_WARNING("Coordinate hit bounds in dim %d, aborting raycast\n", dim);
        // return border point nevertheless:
        end = this->keyToCoord(current_key);
        return false;
      }

      // advance in direction "dim"
      current_key[dim] += step[dim];
      tMax[dim] += tDelta[dim];


      // generate world coords from key
      end = this->keyToCoord(current_key);

      // check for maxrange:
      if (max_range_set){
        double dist_from_origin_sq(0.0);
        for (unsigned int j = 0; j < 3; j++) {
          dist_from_origin_sq += ((end(j) - origin(j)) * (end(j) - origin(j)));
        }
        if (dist_from_origin_sq > maxrange_sq)
          return false;

      }

      NODE* currentNode = this->search(current_key);
      if (currentNode){
        if (this->isNodeOccupied(currentNode)) {
          done = true;
          break;
        }
        // otherwise: node is free and valid, raycasting continues
      } else if (!ignoreUnknown){ // no node found, this usually means we are in "unknown" areas
        OCTOMAP_WARNING_STR("Search failed in OcTree::castRay() => an unknown area was hit in the map: " << end);
        return false;
      }
    } // end while

    return true;
  }


  template <class NODE> inline bool
  OccupancyOcTreeBase<NODE>::integrateMissOnRay(const point3d& origin, const point3d& end, bool lazy_eval) {

    if (!this->computeRayKeys(origin, end, this->keyray)) {
      return false;
    }

    for(KeyRay::iterator it=this->keyray.begin(); it != this->keyray.end(); ++it) {
      updateNode(*it, false, lazy_eval); // insert freespace measurement
    }

    return true;
  }

  template <class NODE> bool
  OccupancyOcTreeBase<NODE>::insertRay(const point3d& origin, const point3d& end, double maxrange, bool lazy_eval)
  {
    // cut ray at maxrange
    if ((maxrange > 0) && ((end - origin).norm () > maxrange))
      {
        point3d direction = (end - origin).normalized ();
        point3d new_end = origin + direction * (float) maxrange;
        return integrateMissOnRay(origin, new_end,lazy_eval);
      }
    // insert complete ray
    else
      {
        if (!integrateMissOnRay(origin, end,lazy_eval))
          return false;
        updateNode(end, true, lazy_eval); // insert hit cell
        return true;
      }
  }

  template <class NODE>
  void OccupancyOcTreeBase<NODE>::getOccupied(point3d_list& node_centers, unsigned int max_depth) const {

    if (max_depth == 0)
      max_depth = this->tree_depth;

    for(typename OccupancyOcTreeBase<NODE>::leaf_iterator it = this->begin(max_depth),
        end=this->end(); it!= end; ++it)
    {
      if(this->isNodeOccupied(*it))
        node_centers.push_back(it.getCoordinate());
    }
  }


  template <class NODE>
  void OccupancyOcTreeBase<NODE>::getOccupied(std::list<OcTreeVolume>& occupied_nodes, unsigned int max_depth) const{

    if (max_depth == 0)
      max_depth = this->tree_depth;

    for(typename OccupancyOcTreeBase<NODE>::leaf_iterator it = this->begin(max_depth),
            end=this->end(); it!= end; ++it)
    {
      if(this->isNodeOccupied(*it))
        occupied_nodes.push_back(OcTreeVolume(it.getCoordinate(), it.getSize()));
    }

  }


  template <class NODE>
  void OccupancyOcTreeBase<NODE>::getOccupied(std::list<OcTreeVolume>& binary_nodes,
                                              std::list<OcTreeVolume>& delta_nodes,
                                              unsigned int max_depth) const{
    if (max_depth == 0)
      max_depth = this->tree_depth;

    for(typename OccupancyOcTreeBase<NODE>::leaf_iterator it = this->begin(max_depth),
            end=this->end(); it!= end; ++it)
    {
      if(this->isNodeOccupied(*it)){
        if (it->getLogOdds() >= this->clamping_thres_max)
          binary_nodes.push_back(OcTreeVolume(it.getCoordinate(), it.getSize()));
        else
          delta_nodes.push_back(OcTreeVolume(it.getCoordinate(), it.getSize()));
      }
    }
  }

  template <class NODE>
  void OccupancyOcTreeBase<NODE>::getFreespace(std::list<OcTreeVolume>& free_nodes, unsigned int max_depth) const{

    if (max_depth == 0)
      max_depth = this->tree_depth;

    for(typename OccupancyOcTreeBase<NODE>::leaf_iterator it = this->begin(max_depth),
            end=this->end(); it!= end; ++it)
    {
      if(!this->isNodeOccupied(*it))
        free_nodes.push_back(OcTreeVolume(it.getCoordinate(), it.getSize()));
    }
  }


  template <class NODE>
  void OccupancyOcTreeBase<NODE>::getFreespace(std::list<OcTreeVolume>& binary_nodes,
                                               std::list<OcTreeVolume>& delta_nodes,
                                               unsigned int max_depth) const{

    if (max_depth == 0)
      max_depth = this->tree_depth;

    for(typename OccupancyOcTreeBase<NODE>::leaf_iterator it = this->begin(max_depth),
            end=this->end(); it!= end; ++it)
    {
      if(!this->isNodeOccupied(*it)){
        if (it->getLogOdds() <= this->clamping_thres_min)
          binary_nodes.push_back(OcTreeVolume(it.getCoordinate(), it.getSize()));
        else
          delta_nodes.push_back(OcTreeVolume(it.getCoordinate(), it.getSize()));
      }
    }
  }


  template <class NODE>
  void OccupancyOcTreeBase<NODE>::setBBXMin (point3d& min) {
    bbx_min = min;
    if (!this->genKey(bbx_min, bbx_min_key)) {
      OCTOMAP_ERROR("ERROR while generating bbx min key.\n");
    }
  }

  template <class NODE>
  void OccupancyOcTreeBase<NODE>::setBBXMax (point3d& max) {
    bbx_max = max;
    if (!this->genKey(bbx_max, bbx_max_key)) {
      OCTOMAP_ERROR("ERROR while generating bbx max key.\n");
    }
  }


  template <class NODE>
  bool OccupancyOcTreeBase<NODE>::inBBX(const point3d& p) const {
    return ((p.x() >= bbx_min.x()) && (p.y() >= bbx_min.y()) && (p.z() >= bbx_min.z()) &&
            (p.x() <= bbx_max.x()) && (p.y() <= bbx_max.y()) && (p.z() <= bbx_max.z()) );
  }


  template <class NODE>
  bool OccupancyOcTreeBase<NODE>::inBBX(const OcTreeKey& key) const {
    return ((key[0] >= bbx_min_key[0]) && (key[1] >= bbx_min_key[1]) && (key[2] >= bbx_min_key[2]) &&
            (key[0] <= bbx_max_key[0]) && (key[1] <= bbx_max_key[1]) && (key[2] <= bbx_max_key[2]) );
  }

  template <class NODE>
  point3d OccupancyOcTreeBase<NODE>::getBBXBounds () const {
    octomap::point3d obj_bounds = (bbx_max - bbx_min);
    obj_bounds /= 2.;
    return obj_bounds;
  }

  template <class NODE>
  point3d OccupancyOcTreeBase<NODE>::getBBXCenter () const {
    octomap::point3d obj_bounds = (bbx_max - bbx_min);
    obj_bounds /= 2.;
    return bbx_min + obj_bounds;
  }


  template <class NODE>
  void OccupancyOcTreeBase<NODE>::getOccupiedLeafsBBX(point3d_list& node_centers, point3d min, point3d max) const {

    OcTreeKey root_key, min_key, max_key;
    root_key[0] = root_key[1] = root_key[2] = this->tree_max_val;
    if (!this->genKey(min, min_key)) return;
    if (!this->genKey(max, max_key)) return;
    getOccupiedLeafsBBXRecurs(node_centers, this->tree_depth, this->root, 0, root_key, min_key, max_key);
  }


  template <class NODE>
  void OccupancyOcTreeBase<NODE>::getOccupiedLeafsBBXRecurs( point3d_list& node_centers, unsigned int max_depth,
                                                             NODE* node, unsigned int depth, const OcTreeKey& parent_key,
                                                             const OcTreeKey& min, const OcTreeKey& max) const {
    if (depth == max_depth) { // max level reached
      if (this->isNodeOccupied(node)) {
        node_centers.push_back(this->keyToCoords(parent_key, depth));
      }
    }

    if (!node->hasChildren()) return;

    unsigned short int center_offset_key = this->tree_max_val >> (depth + 1);

    OcTreeKey child_key;

    for (unsigned int i=0; i<8; ++i) {
      if (node->childExists(i)) {

        computeChildKey(i, center_offset_key, parent_key, child_key);

        // overlap of query bbx and child bbx?
        if (!(
              ( min[0] > (child_key[0] + center_offset_key)) ||
              ( max[0] < (child_key[0] - center_offset_key)) ||
              ( min[1] > (child_key[1] + center_offset_key)) ||
              ( max[1] < (child_key[1] - center_offset_key)) ||
              ( min[2] > (child_key[2] + center_offset_key)) ||
              ( max[2] < (child_key[2] - center_offset_key))
               )) {
          getOccupiedLeafsBBXRecurs(node_centers, max_depth, node->getChild(i), depth+1, child_key, min, max);
        }
      }
    }
  }

  // -- I/O  -----------------------------------------

  template <class NODE>
  std::istream& OccupancyOcTreeBase<NODE>::readBinaryData(std::istream &s){
    this->readBinaryNode(s, this->root);
    this->size_changed = true;
    this->tree_size = OcTreeBaseImpl<NODE,AbstractOccupancyOcTree>::calcNumNodes();  // compute number of nodes
    return s;
  }

  template <class NODE>
  std::ostream& OccupancyOcTreeBase<NODE>::writeBinaryData(std::ostream &s) const{
    OCTOMAP_DEBUG("Writing %u nodes to output stream...", static_cast<unsigned int>(this->size()));
    this->writeBinaryNode(s, this->root);
    return s;
  }

  template <class NODE>
  std::istream& OccupancyOcTreeBase<NODE>::readBinaryNode(std::istream &s, NODE* node) const {

    char child1to4_char;
    char child5to8_char;
    s.read((char*)&child1to4_char, sizeof(char));
    s.read((char*)&child5to8_char, sizeof(char));

    std::bitset<8> child1to4 ((unsigned long long) child1to4_char);
    std::bitset<8> child5to8 ((unsigned long long) child5to8_char);

    //     std::cout << "read:  "
    //        << child1to4.to_string<char,std::char_traits<char>,std::allocator<char> >() << " "
    //        << child5to8.to_string<char,std::char_traits<char>,std::allocator<char> >() << std::endl;


    // inner nodes default to occupied
    node->setLogOdds(this->clamping_thres_max);

    for (unsigned int i=0; i<4; i++) {
      if ((child1to4[i*2] == 1) && (child1to4[i*2+1] == 0)) {
        // child is free leaf
        node->createChild(i);
        node->getChild(i)->setLogOdds(this->clamping_thres_min);
      }
      else if ((child1to4[i*2] == 0) && (child1to4[i*2+1] == 1)) {
        // child is occupied leaf
        node->createChild(i);
        node->getChild(i)->setLogOdds(this->clamping_thres_max);
      }
      else if ((child1to4[i*2] == 1) && (child1to4[i*2+1] == 1)) {
        // child has children
        node->createChild(i);
        node->getChild(i)->setLogOdds(-200.); // child is unkown, we leave it uninitialized
      }
    }
    for (unsigned int i=0; i<4; i++) {
      if ((child5to8[i*2] == 1) && (child5to8[i*2+1] == 0)) {
        // child is free leaf
        node->createChild(i+4);
        node->getChild(i+4)->setLogOdds(this->clamping_thres_min);
      }
      else if ((child5to8[i*2] == 0) && (child5to8[i*2+1] == 1)) {
        // child is occupied leaf
        node->createChild(i+4);
        node->getChild(i+4)->setLogOdds(this->clamping_thres_max);
      }
      else if ((child5to8[i*2] == 1) && (child5to8[i*2+1] == 1)) {
        // child has children
        node->createChild(i+4);
        node->getChild(i+4)->setLogOdds(-200.); // set occupancy when all children have been read
      }
      // child is unkown, we leave it uninitialized
    }

    // read children's children and set the label
    for (unsigned int i=0; i<8; i++) {
      if (node->childExists(i)) {
        NODE* child = node->getChild(i);
        if (fabs(child->getLogOdds() + 200.)<1e-3) {
          readBinaryNode(s, child);
          child->setLogOdds(child->getMaxChildLogOdds());
        }
      } // end if child exists
    } // end for children

    return s;
  }

  template <class NODE>
  std::ostream& OccupancyOcTreeBase<NODE>::writeBinaryNode(std::ostream &s, const NODE* node) const{

    // 2 bits for each children, 8 children per node -> 16 bits
    std::bitset<8> child1to4;
    std::bitset<8> child5to8;

    // 10 : child is free node
    // 01 : child is occupied node
    // 00 : child is unkown node
    // 11 : child has children


    // speedup: only set bits to 1, rest is init with 0 anyway,
    //          can be one logic expression per bit

    for (unsigned int i=0; i<4; i++) {
      if (node->childExists(i)) {
        const NODE* child = node->getChild(i);
        if      (child->hasChildren())  { child1to4[i*2] = 1; child1to4[i*2+1] = 1; }
        else if (this->isNodeOccupied(child)) { child1to4[i*2] = 0; child1to4[i*2+1] = 1; }
        else                            { child1to4[i*2] = 1; child1to4[i*2+1] = 0; }
      }
      else {
        child1to4[i*2] = 0; child1to4[i*2+1] = 0;
      }
    }

    for (unsigned int i=0; i<4; i++) {
      if (node->childExists(i+4)) {
        const NODE* child = node->getChild(i+4);
        if      (child->hasChildren())  { child5to8[i*2] = 1; child5to8[i*2+1] = 1; }
        else if (this->isNodeOccupied(child)) { child5to8[i*2] = 0; child5to8[i*2+1] = 1; }
        else                            { child5to8[i*2] = 1; child5to8[i*2+1] = 0; }
      }
      else {
        child5to8[i*2] = 0; child5to8[i*2+1] = 0;
      }
    }
    //     std::cout << "wrote: "
    //        << child1to4.to_string<char,std::char_traits<char>,std::allocator<char> >() << " "
    //        << child5to8.to_string<char,std::char_traits<char>,std::allocator<char> >() << std::endl;

    char child1to4_char = (char) child1to4.to_ulong();
    char child5to8_char = (char) child5to8.to_ulong();

    s.write((char*)&child1to4_char, sizeof(char));
    s.write((char*)&child5to8_char, sizeof(char));

    // write children's children
    for (unsigned int i=0; i<8; i++) {
      if (node->childExists(i)) {
        const NODE* child = node->getChild(i);
        if (child->hasChildren()) {
          writeBinaryNode(s, child);
        }
      }
    }

    return s;
  }

  //-- Occupancy queries on nodes:

  template <class NODE>
  void OccupancyOcTreeBase<NODE>::updateNodeLogOdds(NODE* occupancyNode, const float& update) const {
    occupancyNode->addValue(update);
    if (occupancyNode->getLogOdds() < this->clamping_thres_min) {
      occupancyNode->setLogOdds(this->clamping_thres_min);
      return;
    }
    if (occupancyNode->getLogOdds() > this->clamping_thres_max) {
      occupancyNode->setLogOdds(this->clamping_thres_max);
    }
  }

  template <class NODE>
  void OccupancyOcTreeBase<NODE>::integrateHit(NODE* occupancyNode) const {
    updateNodeLogOdds(occupancyNode, this->prob_hit_log);
  }

  template <class NODE>
  void OccupancyOcTreeBase<NODE>::integrateMiss(NODE* occupancyNode) const {
    updateNodeLogOdds(occupancyNode, this->prob_miss_log);
  }

  template <class NODE>
  void OccupancyOcTreeBase<NODE>::nodeToMaxLikelihood(NODE* occupancyNode) const{
    if (this->isNodeOccupied(occupancyNode))
      occupancyNode->setLogOdds(this->clamping_thres_max);
    else
      occupancyNode->setLogOdds(this->clamping_thres_min);
  }

  template <class NODE>
  void OccupancyOcTreeBase<NODE>::nodeToMaxLikelihood(NODE& occupancyNode) const{
    if (this->isNodeOccupied(occupancyNode))
      occupancyNode.setLogOdds(this->clamping_thres_max);
    else
      occupancyNode.setLogOdds(this->clamping_thres_min);
  }

} // namespace
